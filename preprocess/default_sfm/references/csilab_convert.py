"""Convert trusted demo pickles into one HDF5 file per episode using YAML settings."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from tools.demo_viz.schema import SchemaError, state_layout, state_width, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).parent / 'configs/config.yaml'


class PreprocessError(ValueError):
    pass


ROTATION_COLUMNS = {
    'rpy': ['roll', 'pitch', 'yaw'],
    'gravity': ['gravity_x', 'gravity_y', 'gravity_z'],
    'quaternion': ['qw', 'qx', 'qy', 'qz'],
    'rotation_6d': ['r00', 'r10', 'r20', 'r01', 'r11', 'r21'],
    'rotation_matrix': [f'r{i}{j}' for i in range(3) for j in range(3)],
}


def state_rotation_representation(item):
    rep = 'rpy' if isinstance(item, str) else item.get('rotation_representation', 'rpy')
    if not isinstance(rep, str) or rep not in ROTATION_COLUMNS:
        raise PreprocessError('rotation_representation must be rpy, gravity, quaternion, rotation_6d or rotation_matrix')
    if rep != 'rpy' and item.get('key', '').split('/')[-1] != 'tcp_pose':
        raise PreprocessError('rotation_representation is supported only for tcp_pose observations')
    return rep


def rotation_observation(rpy, representation):
    rotation = Rotation.from_euler('xyz', rpy)
    if representation == 'gravity':
        return rotation.inv().apply([0., 0., -1.])
    if representation == 'quaternion':
        return rotation.as_quat(canonical=True, scalar_first=True)
    matrix = rotation.as_matrix()
    return matrix[:, :2].T.reshape(-1) if representation == 'rotation_6d' else matrix.reshape(-1)


def fail_unknown(mapping, allowed, name):
    if not isinstance(mapping, dict):
        raise PreprocessError(f'{name} must be a mapping')
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise PreprocessError(f'{name}: unknown settings {sorted(unknown)}')


def state_wrap_angles(item):
    """Euler observation components to map to [0, 2*pi), independently per sample."""
    if isinstance(item, str):
        return []
    wrapped = item.get('wrap_angles', [])
    if (not isinstance(wrapped, list)
            or any(not isinstance(name, str) or name not in ('roll', 'pitch', 'yaw') for name in wrapped)
            or len(wrapped) != len(set(wrapped))):
        raise PreprocessError('wrap_angles must list unique Euler component names: roll, pitch, yaw')
    if wrapped and item.get('key', '').split('/')[-1] != 'tcp_pose':
        raise PreprocessError('wrap_angles is supported only for tcp_pose Euler observations')
    if wrapped and state_rotation_representation(item) != 'rpy':
        raise PreprocessError('wrap_angles cannot be combined with a non-rpy rotation_representation')
    return wrapped


def validate_config(cfg):
    fail_unknown(cfg, ('version', 'input', 'output', 'observations', 'crop_initial_stationary', 'actions', 'normalization'), 'config')
    if cfg.get('version') != 1:
        raise PreprocessError('config.version must be 1')
    fail_unknown(cfg.get('input'), ('path', 'recursive', 'collection_config'), 'input')
    fail_unknown(cfg.get('output'), ('directory', 'overwrite', 'compression', 'save_next_observations'), 'output')
    if (not isinstance(cfg['input'].get('path'), str) or not cfg['input']['path'] or
            not isinstance(cfg['output'].get('directory'), str) or not cfg['output']['directory']):
        raise PreprocessError('input.path and output.directory are required')
    if cfg['input'].get('collection_config') is not None and not isinstance(cfg['input']['collection_config'], str):
        raise PreprocessError('input.collection_config must be a path string or null')
    for section, name in [('input', 'recursive'), ('output', 'overwrite'), ('output', 'save_next_observations')]:
        if name in cfg[section] and not isinstance(cfg[section][name], bool):
            raise PreprocessError(f'{section}.{name} must be true/false')
    if cfg['output'].get('compression', 'lzf') not in ('lzf', 'gzip', None):
        raise PreprocessError('output.compression must be lzf, gzip, or null')
    from .action_chunks import validate as validate_actions
    from .normalization import validate as validate_normalization
    if 'actions' in cfg:
        validate_actions(cfg['actions'])
    validate_normalization(cfg.get('normalization', {}))
    action_normalization = cfg.get('normalization', {}).get('action', {})
    from .normalization import group_options, ACTION_GROUPS
    if 'actions' not in cfg and (action_normalization.get('method', 'none') != 'none' or
            any(group_options(action_normalization, group).get('method', 'none') != 'none' for group in ACTION_GROUPS)):
        raise PreprocessError('Action normalization requires the actions chunk configuration')
    entries = cfg.get('observations')
    if not isinstance(entries, list) or not entries:
        raise PreprocessError('observations must be a nonempty list')
    targets = set()
    for entry in entries:
        kind = entry.get('type') if isinstance(entry, dict) else None
        options = {'state': ('keys',), 'image': ('input_color', 'crop', 'rotation_deg', 'rotation_expand',
                                             'resize', 'resize_mode', 'interpolation'),
                   'array': ('sample_ndim',)}
        if kind not in options:
            raise PreprocessError(f'Unknown observation type: {kind}')
        fail_unknown(entry, ('source', 'target', 'type') + options[kind], 'observation')
        source, target = entry.get('source'), entry.get('target')
        if not isinstance(source, str) or not source:
            raise PreprocessError('Every observation needs a source key')
        if not isinstance(target, str) or not target or '/' in target or target in ('.', '..'):
            raise PreprocessError('target must be a nonempty HDF5 key without slashes')
        if target in targets:
            raise PreprocessError(f'Duplicate observation target: {target}')
        targets.add(target)
        if kind == 'state':
            keys = entry.get('keys')
            if not isinstance(keys, list) or not keys:
                raise PreprocessError('state.keys must be a nonempty list')
            names = []
            for item in keys:
                if isinstance(item, str):
                    names.append(item)
                else:
                    fail_unknown(item, ('key', 'components', 'wrap_angles', 'rotation_representation'), 'state selection')
                    if not isinstance(item.get('key'), str) or not isinstance(item.get('components'), (list, dict)):
                        raise PreprocessError('state selection needs key and components list/boolean mapping')
                    state_wrap_angles(item)
                    state_rotation_representation(item)
                    names.append(item['key'])
            if len(names) != len(set(names)):
                raise PreprocessError('state.keys must not contain duplicate channels')
        elif kind == 'array':
            if type(entry.get('sample_ndim')) is not int or entry['sample_ndim'] < 1:
                raise PreprocessError('array.sample_ndim must be a positive integer')
        else:
            if entry.get('input_color', 'rgb') not in ('rgb', 'bgr'):
                raise PreprocessError('image.input_color must be rgb/bgr')
            if entry.get('resize_mode', 'stretch') not in ('stretch', 'letterbox'):
                raise PreprocessError('image.resize_mode must be stretch/letterbox')
            if entry.get('interpolation', 'area') not in ('area', 'linear', 'nearest', 'cubic'):
                raise PreprocessError('Unsupported image interpolation')
            angle = entry.get('rotation_deg', 0)
            if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not np.isfinite(angle):
                raise PreprocessError('image.rotation_deg must be a finite number (positive = counterclockwise)')
            if not isinstance(entry.get('rotation_expand', True), bool):
                raise PreprocessError('image.rotation_expand must be true/false')
            for key, count in [('resize', 2), ('crop', 4)]:
                value = entry.get(key)
                if value is not None and (not isinstance(value, list) or len(value) != count or
                    not all(type(v) is int for v in value) or any(v <= 0 for v in (value if key == 'resize' else value[2:])) or
                    (key == 'crop' and any(v < 0 for v in value[:2]))):
                    raise PreprocessError(f'Invalid image {key}: {value}')
    crop = cfg.get('crop_initial_stationary', {})
    fail_unknown(crop, ('enabled', 'pose_keys', 'gripper_keys', 'translation_threshold_m', 'rotation_threshold_rad',
                        'gripper_threshold', 'consecutive_frames', 'pre_roll_frames', 'no_motion'), 'crop_initial_stationary')
    if not isinstance(crop.get('enabled', False), bool):
        raise PreprocessError('crop.enabled must be true/false')
    if crop.get('no_motion', 'keep') not in ('keep', 'drop', 'error'):
        raise PreprocessError('crop.no_motion must be keep/drop/error')
    for key, default, minimum in [('consecutive_frames', 3, 1), ('pre_roll_frames', 2, 0)]:
        value = crop.get(key, default)
        if type(value) is not int or value < minimum:
            raise PreprocessError(f'crop.{key} must be an integer >= {minimum}')
    for key, default in [('translation_threshold_m', .001), ('rotation_threshold_rad', .01), ('gripper_threshold', .02)]:
        value = crop.get(key, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise PreprocessError(f'crop.{key} must be finite and nonnegative')
    for key in ('pose_keys', 'gripper_keys'):
        if not isinstance(crop.get(key, []), list) or not all(isinstance(k, str) for k in crop.get(key, [])):
            raise PreprocessError(f'crop.{key} must be a list of state keys')
    if crop.get('enabled', False) and not (crop.get('pose_keys') or crop.get('gripper_keys')):
        raise PreprocessError('Enabled stationary cropping needs pose_keys and/or gripper_keys')


def read_settings(path):
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    validate_config(cfg)
    return cfg, path


def setting_path(value, config_path):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (config_path.parent / p).resolve()


def latest(value, ndim):
    array = np.asarray(value)
    if array.ndim == ndim + 1:
        if not array.shape[0]:
            raise PreprocessError('Empty observation horizon')
        array = array[-1]
    if array.ndim != ndim:
        raise PreprocessError(f'Expected {ndim} sample dimensions with optional horizon, got {array.shape}')
    return array


def numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'biuf' or not np.all(np.isfinite(array)):
        raise PreprocessError(f'{name}: expected finite numeric values, got dtype={array.dtype}')
    return array


def terminal(row):
    return bool(row.get('dones', False) or row.get('terminated', False) or row.get('truncated', False))


def episodes(data):
    """Split boundary-aware; a later episode is never used as next_observations."""
    structured = isinstance(data, dict) and 'transitions' in data
    rows = data['transitions'] if structured else data
    if not isinstance(rows, (list, tuple)):
        raise PreprocessError('Expected a transition list or a dictionary containing transitions')
    metadata = data.get('metadata', {}) if structured else {}
    current = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get('observations'), dict):
            raise PreprocessError(f'Transition {index} has no observation dictionary')
        current.append(row)
        if terminal(row) or index == len(rows) - 1:
            final = data.get('final_observation') if structured and index == len(rows) - 1 else None
            yield current, metadata, final
            current = []


def collection_config(source, metadata, explicit=None):
    if explicit:
        return load_config(explicit), str(explicit)
    if isinstance(metadata.get('config'), dict):
        return metadata['config'], 'embedded'
    for parent in source.parents:
        path = ROOT / 'Real_Robo/configs' / (parent.name + '.json')
        if path.is_file():
            return load_config(path), str(path)
    raise PreprocessError(f'Cannot decode state columns for {source}; set input.collection_config')


class StateDecoder:
    def __init__(self, collection):
        self.gripper_interfaces = collection.get('gripper_interfaces', {})
        self.layout = state_layout(collection)
        self.width = state_width(self.layout)
        self.lookup = {e['key']: e for e in self.layout}
        if not self.width:
            raise PreprocessError('Collection config needs train.proprio_keys')

    def channel(self, observation, key, source='state'):
        if key not in self.lookup:
            raise PreprocessError(f'State key {key!r} absent from collection config')
        entry = self.lookup[key]
        if source not in observation:
            raise PreprocessError(f'Missing observation source: {source}')
        state = observation[source]
        if isinstance(state, dict):
            if key not in state:
                raise PreprocessError(f'Missing state channel: {key}')
            values = latest(state[key], 1)
            if values.size != entry['width']:
                raise PreprocessError(f'{key}: expected {entry["width"]} values, got {values.size}')
        else:
            state = latest(state, 1)
            if state.size != self.width:
                raise PreprocessError(f'State width {state.size} != collection config width {self.width}')
            values = state[entry['offset']:entry['offset'] + entry['width']]
        return numeric(values, key)

    def selection(self, item):
        key = item if isinstance(item, str) else item['key']
        if key not in self.lookup:
            raise PreprocessError(f'Unknown state key: {key}')
        names = self.lookup[key]['components']
        components = names if isinstance(item, str) else item['components']
        if isinstance(components, dict):
            if set(components) - set(names) or not all(type(v) is bool for v in components.values()):
                raise PreprocessError(f'{key}: components must map known component names to true/false')
            components = [name for name in names if components.get(name, False)]
        indices = []
        for component in components:
            if type(component) is int and 0 <= component < len(names):
                indices.append(component)
            elif isinstance(component, str) and component in names:
                indices.append(names.index(component))
            else:
                raise PreprocessError(f'{key}: invalid component {component!r}; expected {names}')
        if len(indices) != len(set(indices)):
            raise PreprocessError(f'{key}: duplicate components')
        wrapped = state_wrap_angles(item)
        rep = state_rotation_representation(item)
        if rep != 'rpy' and not {3, 4, 5}.issubset(indices):
            raise PreprocessError(f'{key}: {rep} requires all roll, pitch and yaw components; use the full rotation block')
        if set(wrapped) - {names[index] for index in indices}:
            raise PreprocessError(f'{key}: wrap_angles must refer to retained rotation components')
        return key, indices

    def selected(self, observation, item, source):
        key, indices = self.selection(item)
        raw = self.channel(observation, key, source)
        side, _, channel = key.partition('/')
        if channel == 'gripper_pose' and side in self.gripper_interfaces:
            from rb_env.gripper_interface import GripperInterface
            raw = GripperInterface.from_contract(self.gripper_interfaces[side]).observed_opening(raw)
        rep = state_rotation_representation(item)
        if rep != 'rpy':
            encoded = rotation_observation(raw[3:6], rep)
            values, emitted = [], False
            for i in indices:
                if i < 3:
                    values.append(raw[i:i + 1])
                elif not emitted:
                    values.append(encoded)
                    emitted = True
            return np.concatenate(values)
        values = raw[indices]
        wrapped = state_wrap_angles(item)
        if wrapped:
            values = values.astype(np.float64)
            # The final HDF5/model state is float32. Keep its upper bound open
            # even when a tiny negative angle would round up to float32(2*pi).
            upper = float(np.nextafter(np.float32(2 * np.pi), np.float32(0)))
            for output_index, channel_index in enumerate(indices):
                if self.lookup[key]['components'][channel_index] in wrapped:
                    values[output_index] = min(float(np.remainder(values[output_index], 2 * np.pi)), upper)
        return values

    def columns(self, keys):
        columns = []
        for item in keys:
            key, indices = self.selection(item)
            rep = state_rotation_representation(item)
            emitted = False
            for i in indices:
                if rep == 'rpy' or i < 3:
                    columns.append(key + '/' + self.lookup[key]['components'][i])
                elif not emitted:
                    columns.extend(key + '/' + name for name in ROTATION_COLUMNS[rep])
                    emitted = True
        if not columns:
            raise PreprocessError('State selection must retain at least one component')
        return columns


def crop_start(rows, decoder, cfg):
    result = {'enabled': cfg.get('enabled', False), 'original_frames': len(rows), 'start': 0,
              'detected_motion_frame': None, 'reason': 'disabled'}
    if not result['enabled']:
        return 0, result
    moving = np.zeros(len(rows), dtype=bool)
    for key in cfg.get('pose_keys', []):
        poses = np.stack([decoder.channel(row['observations'], key) for row in rows])
        if poses.shape[1] != 6:
            raise PreprocessError(f'Cropping pose {key} must contain xyz + XYZ Euler (m/rad)')
        position = np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1)
        rotations = Rotation.from_euler('xyz', poses[:, 3:6])
        angles = (rotations[0].inv() * rotations).magnitude()
        moving |= (position > cfg.get('translation_threshold_m', .001)) | (angles > cfg.get('rotation_threshold_rad', .01))
    for key in cfg.get('gripper_keys', []):
        grip = np.stack([decoder.channel(row['observations'], key) for row in rows])
        if grip.shape[1] != 1:
            raise PreprocessError(f'Cropping gripper {key} must be scalar')
        moving |= np.abs(grip[:, 0] - grip[0, 0]) > cfg.get('gripper_threshold', .02)
    run = 0
    for index, is_moving in enumerate(moving):
        run = run + 1 if is_moving else 0
        if run >= cfg.get('consecutive_frames', 3):
            first = index - run + 1
            start = max(0, first - cfg.get('pre_roll_frames', 2))
            result.update(start=start, detected_motion_frame=first, reason='motion')
            return start, result
    policy = cfg.get('no_motion', 'keep')
    result['reason'] = 'no_motion_' + policy
    if policy == 'error':
        raise PreprocessError('No sustained motion found in episode')
    if policy == 'drop':
        result['start'] = len(rows)
        return None, result
    return 0, result


def image_sample(value, entry):
    frame = latest(value, 3)
    if frame.dtype != np.uint8 or frame.shape[-1] != 3:
        raise PreprocessError(f'Image must be uint8 HWC RGB/BGR, got {frame.shape} {frame.dtype}')
    if entry.get('input_color', 'rgb') == 'bgr':
        frame = frame[..., ::-1]
    if entry.get('crop') is not None:
        top, left, height, width = entry['crop']
        if top + height > frame.shape[0] or left + width > frame.shape[1]:
            raise PreprocessError(f'Image crop {entry["crop"]} exceeds frame {frame.shape}')
        frame = frame[top:top + height, left:left + width]
    # Keep crop coordinates in the original frame. Positive angles are CCW.
    angle = float(entry.get('rotation_deg', 0)) % 360
    if angle:
        height, width = frame.shape[:2]
        expand = entry.get('rotation_expand', True)
        if angle % 90 == 0 and (expand or angle == 180 or height == width):
            # Exact quarter turns preserve pixels and avoid floating-point border artifacts.
            frame = np.rot90(frame, int(angle / 90))
        else:
            matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.)
            out_h, out_w = height, width
            if expand:
                cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
                out_w = int(np.ceil(width * cosine + height * sine))
                out_h = int(np.ceil(height * cosine + width * sine))
                matrix[0, 2] += (out_w - width) / 2
                matrix[1, 2] += (out_h - height) / 2
            # AREA is a resize filter; arbitrary-angle rotation uses LINEAR in that case.
            interpolation = {'area': cv2.INTER_LINEAR, 'linear': cv2.INTER_LINEAR,
                             'nearest': cv2.INTER_NEAREST, 'cubic': cv2.INTER_CUBIC}[entry.get('interpolation', 'area')]
            frame = cv2.warpAffine(np.ascontiguousarray(frame), matrix, (out_w, out_h),
                                   flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if entry.get('resize') is not None:
        h, w = entry['resize']
        interpolation = {'area': cv2.INTER_AREA, 'linear': cv2.INTER_LINEAR,
                         'nearest': cv2.INTER_NEAREST, 'cubic': cv2.INTER_CUBIC}[entry.get('interpolation', 'area')]
        if entry.get('resize_mode', 'stretch') == 'letterbox':
            scale = min(h / frame.shape[0], w / frame.shape[1])
            rh, rw = max(1, round(frame.shape[0] * scale)), max(1, round(frame.shape[1] * scale))
            resized = cv2.resize(frame, (rw, rh), interpolation=interpolation)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            top, left = (h - rh) // 2, (w - rw) // 2
            frame[top:top + rh, left:left + rw] = resized
        elif frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=interpolation)
    return np.ascontiguousarray(frame)


def transform_observation(observation, entries, decoder):
    transformed = {}
    for entry in entries:
        source, target = entry['source'], entry['target']
        if source not in observation:
            raise PreprocessError(f'Missing observation source {source!r} for target {target!r}')
        if entry['type'] == 'state':
            decoder.columns(entry['keys'])  # Validate dimensions before writing.
            value = np.concatenate([decoder.selected(observation, item, source) for item in entry['keys']])
        elif entry['type'] == 'image':
            value = image_sample(observation[source], entry)
        else:
            value = latest(observation[source], entry['sample_ndim'])
        if entry['type'] != 'image':
            value = numeric(value, source).astype(np.float32)
        transformed[target] = value
    return transformed


def next_observation(rows, index, final):
    row = rows[index]
    if isinstance(row.get('next_observations'), dict):
        return row['next_observations']
    if index + 1 < len(rows):
        return rows[index + 1]['observations']
    if isinstance(final, dict):
        return final
    raise PreprocessError('Final next_observations is missing; supply final_observation '
                          '(required by next-observation output, observed pose labels or next-state gripper labels)')


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def store_json(handle, key, value):
    handle.create_dataset(key, data=json.dumps(value, ensure_ascii=False, default=json_default), dtype=h5py.string_dtype('utf-8'))


def write_episode(rows, final, metadata, collection, cfg, source, output, episode_index, config_path, collection_source, dry_run=False):
    decoder = StateDecoder(collection)
    start, crop = crop_start(rows, decoder, cfg.get('crop_initial_stationary', {}))
    summary = dict(source=str(source), episode=episode_index, output=str(output), **crop)
    if start is None:
        return dict(summary, frames=0, skipped=True)
    count = len(rows) - start
    summary.update(frames=count, skipped=False)
    entries = cfg['observations']
    sample = transform_observation(rows[start]['observations'], entries, decoder)
    summary['observations'] = {k: [count, *v.shape] for k, v in sample.items()}
    action = numeric(rows[start]['actions'], 'actions')
    if action.ndim < 1 or not action.size:
        raise PreprocessError('actions must have at least one nonempty dimension')
    from .action_chunks import ActionChunks
    chunks = ActionChunks(rows, metadata, collection, decoder, cfg['actions'], final=final) if 'actions' in cfg else None
    if chunks and chunks.gripper is not None and chunks.gripper.settings.get('label_version') == 2:
        collection = copy.deepcopy(collection)
        collection['gripper_interfaces'] = {side: interface.contract() for side, interface in zip(chunks.sides, chunks.gripper.interfaces)}
        decoder = StateDecoder(collection)
    summary['actions'] = [count, chunks.length, chunks.action_dim] if chunks else [count, *action.shape]
    if chunks:
        summary['chunk_mode'] = chunks.mode
        if chunks.gripper is not None:
            summary['gripper'] = dict(mode=chunks.gripper.settings['mode'],
                                      grasp_frames=chunks.gripper.grasp[start:].sum(axis=0).tolist(), sides=chunks.sides)
    if dry_run:
        return summary
    overwrite = cfg['output'].get('overwrite', False)
    if output.exists() and not overwrite:
        raise PreprocessError(f'Output exists: {output}; choose another output directory or --overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.stem + '.', suffix='.tmp', dir=output.parent)
    os.close(fd)
    temporary = Path(temporary)
    compression = cfg['output'].get('compression', 'lzf')
    datasets = {}
    formats = set()
    try:
        with h5py.File(temporary, 'w') as handle:
            handle.attrs.update(schema_version=1, source_file=str(source), episode_index=episode_index,
                                num_steps=count, original_num_steps=len(rows), crop_start=start,
                                observation_horizon='latest', image_color='RGB', action_transform='none',
                                normalization='none', time_semantics='source transition timestamp; NaN if absent')
            store_json(handle, 'metadata/preprocess_config', cfg)
            store_json(handle, 'metadata/collection_config', collection)
            store_json(handle, 'metadata/recording_metadata', metadata)
            store_json(handle, 'metadata/crop', crop)
            if chunks:
                store_json(handle, 'metadata/action_contract', chunks.contract())
                handle.attrs['action_transform'] = chunks.mode
                handle.attrs['schema_version'] = 2
            handle.attrs['preprocess_config_path'] = str(config_path)
            handle.attrs['collection_config_source'] = collection_source
            for out_index, source_index in enumerate(range(start, len(rows))):
                row = rows[source_index]
                values = {'observations/' + k: v for k, v in transform_observation(row['observations'], entries, decoder).items()}
                if cfg['output'].get('save_next_observations', True):
                    values.update({'next_observations/' + k: v for k, v in transform_observation(next_observation(rows, source_index, final), entries, decoder).items()})
                actions = numeric(row['actions'], 'actions')  # Keep original action dtype, dimensions and units.
                if actions.shape != action.shape or actions.dtype != action.dtype:
                    raise PreprocessError(f'Action shape/dtype changes at source frame {source_index}')
                if chunks:
                    values['source_actions'] = actions
                    if chunks.delta is not None:
                        values['delta_actions'] = chunks.delta[source_index]
                    values['action_reference_poses'] = chunks.poses[source_index]
                    if chunks.gripper is not None:
                        for key, array in [('gripper_commands', chunks.gripper.commands),
                                           ('gripper_previous_commands', chunks.gripper.previous),
                                           ('gripper_observed', chunks.gripper.observed),
                                           ('gripper_next_observed', chunks.gripper.next_observed),
                                           ('gripper_labels', chunks.gripper.labels),
                                           ('gripper_grasp_mask', chunks.gripper.grasp)]:
                            if array is not None:
                                values[key] = array[source_index]
                        if hasattr(chunks.gripper, 'observation_valid'):
                            values['gripper_observation_valid'] = chunks.gripper.observation_valid[source_index]
                            values['gripper_opening'] = chunks.gripper.opening[source_index]
                    actions, values['action_mask'] = chunks.at(source_index)
                info = row.get('infos', {}) or {}
                formats.add(info.get('action_format') or metadata.get('command_format') or 'unspecified')
                timestamp = row.get('time', (row.get('action_context') or {}).get('timestamp'))
                is_terminal = terminal(row)
                is_truncated = bool(row.get('truncated', False))
                is_terminated = bool(row.get('terminated', is_terminal and not is_truncated))
                values.update(actions=actions, rewards=np.asarray(numeric(row.get('rewards', 0), 'rewards'), dtype=np.float32),
                              terminals=np.asarray(is_terminal, dtype=np.bool_),
                              terminated=np.asarray(is_terminated, dtype=np.bool_),
                              truncated=np.asarray(is_truncated, dtype=np.bool_),
                              masks=np.asarray(numeric(row.get('masks', 1.0 - is_terminated), 'masks'), dtype=np.float32),
                              timestamps=np.asarray(np.nan if timestamp is None else timestamp, dtype=np.float64),
                              source_indices=np.asarray(source_index, dtype=np.int64))
                for key, value in values.items():
                    value = np.asarray(value)
                    if out_index == 0:
                        dataset = handle.create_dataset(key, shape=(count, *value.shape), dtype=value.dtype,
                                                        chunks=True, compression=compression)
                        datasets[key] = dataset
                    else:
                        dataset = datasets[key]
                        if dataset.shape[1:] != value.shape or dataset.dtype != value.dtype:
                            raise PreprocessError(f'{key}: shape/dtype changes at source frame {source_index}')
                    dataset[out_index] = value
            for group in ('observations', 'next_observations'):
                if group not in handle:
                    continue
                for entry in entries:
                    dataset = handle[group + '/' + entry['target']]
                    dataset.attrs['source_key'] = entry['source']
                    dataset.attrs['type'] = entry['type']
                    if entry['type'] == 'state':
                        dataset.attrs['columns'] = json.dumps(decoder.columns(entry['keys']))
                    elif entry['type'] == 'image':
                        dataset.attrs['color_order'] = 'RGB'
                        dataset.attrs['layout'] = 'NHWC'
            handle['actions'].attrs['format'] = next(iter(formats)) if len(formats) == 1 else 'mixed'
            handle['actions'].attrs['source_key'] = 'actions'
            handle['actions'].attrs['transform'] = chunks.mode if chunks else 'none'
            if chunks:
                handle['actions'].attrs['format'] = {'delta': 'world_delta_xyz_rotvec_gripper',
                                                     'delta_flange': 'flange_delta_xyz_rotvec_gripper',
                                                     'relative': 'relative_tcp_xyz_rotvec_gripper',
                                                     'relative_observed': 'relative_tcp_observed_xyz_rotvec_gripper',
                                                     'relative_observed_residual': 'relative_tcp_observed_residual_xyz_rotvec_gripper'}[chunks.mode]
                handle['actions'].attrs['columns'] = json.dumps(chunks.contract()['columns'])
                handle['actions'].attrs['padding_mask'] = 'action_mask'
                handle['source_actions'].attrs['format'] = next(iter(formats)) if len(formats) == 1 else 'mixed'
                if 'delta_actions' in handle:
                    handle['delta_actions'].attrs['normalization'] = 'none'
                handle['action_reference_poses'].attrs['format'] = 'xyz_wxyz, one pose per arm in actions.sides order'
        if overwrite:
            os.replace(temporary, output)
        else:
            # Atomic publication, refusing concurrent overwrites as well.
            os.link(temporary, output)
        return summary
    finally:
        temporary.unlink(missing_ok=True)


def _run_raw(cfg, config_path, input_path=None, output_dir=None, dry_run=False, quiet=False):
    validate_config(cfg)
    source = Path(input_path).expanduser().resolve() if input_path else setting_path(cfg['input']['path'], config_path)
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else setting_path(cfg['output']['directory'], config_path)
    if source.is_dir():
        files = sorted(source.rglob('*.pkl') if cfg['input'].get('recursive', True) else source.glob('*.pkl'))
    elif source.is_file() and source.suffix == '.pkl':
        files = [source]
    else:
        raise PreprocessError(f'Input pickle/folder not found: {source}')
    if not files:
        raise PreprocessError(f'No pickle files in {source}')
    explicit = cfg['input'].get('collection_config')
    explicit = setting_path(explicit, config_path) if explicit else None
    summaries = []
    for path in files:
        before = path.stat()
        try:
            with path.open('rb') as stream:
                data = pickle.load(stream)
        except (EOFError, pickle.UnpicklingError) as exc:
            raise PreprocessError(f'Incomplete or invalid pickle: {path}; wait for recording to finish') from exc
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise PreprocessError(f'File changed during read: {path}')
        for index, (rows, metadata, final) in enumerate(episodes(data)):
            collection, collection_source = collection_config(path.resolve(), metadata, explicit)
            source_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]
            output = output_dir / f'{path.stem}__{source_id}__ep{index:04d}.hdf5'
            summary = write_episode(rows, final, metadata, collection, cfg, path.resolve(), output, index,
                                    config_path, collection_source, dry_run=dry_run)
            summaries.append(summary)
            if not quiet:
                print(json.dumps(summary, ensure_ascii=False), flush=True)
        del data
    if not summaries:
        raise PreprocessError('Input contains no episodes')
    return summaries


def run(cfg, config_path, input_path=None, output_dir=None, dry_run=False):
    from .normalization import fit_or_load, apply_file
    validate_config(cfg)
    cfg = copy.deepcopy(cfg)
    source = Path(input_path).expanduser().resolve() if input_path else setting_path(cfg['input']['path'], config_path)
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else setting_path(cfg['output']['directory'], config_path)
    cfg['input']['path'] = str(source)
    cfg['output']['directory'] = str(output_dir)
    if cfg['input'].get('collection_config'):
        cfg['input']['collection_config'] = str(setting_path(cfg['input']['collection_config'], config_path))
    if cfg.get('normalization', {}).get('stats_file'):
        cfg['normalization']['stats_file'] = str(setting_path(cfg['normalization']['stats_file'], config_path))
    if dry_run:
        return _run_raw(cfg, config_path, source, output_dir, dry_run=True)
    if output_dir.exists() and (list(output_dir.glob('*.hdf5')) or (output_dir/'config.yaml').exists()) and not cfg['output'].get('overwrite', False):
        raise PreprocessError(f'Output exists: {output_dir}; use a fresh directory or --overwrite for the whole dataset')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.preprocess-', dir=output_dir.parent) as directory:
        stage = Path(directory)
        summaries = _run_raw(cfg, config_path, source, stage, quiet=True)
        paths = [Path(s['output']) for s in summaries if not s['skipped']]
        if not paths:
            return summaries
        stats_file = cfg.get('normalization', {}).get('stats_file')
        stats_file = setting_path(stats_file, config_path) if stats_file else None
        stats = fit_or_load(paths, cfg, stats_file)
        with h5py.File(paths[0]) as f:
            collection = json.loads(f['metadata/collection_config'][()])
            contract = json.loads(f['metadata/action_contract'][()]) if 'metadata/action_contract' in f else {'mode': 'raw'}
        inference = dict(version=1, observations=cfg['observations'], collection_config=collection,
                         actions=contract, normalization_file='normalization.json',
                         action_period_s=1. / float(collection.get('train', {}).get('hz', 10)))
        env_config = copy.deepcopy(collection)
        if 'interpreter_mode' in contract:
            env_config.setdefault('train', {})['action_chunk_mode'] = contract['interpreter_mode']
        # Complete and validate every file before publishing any dataset changes.
        for path in paths:
            apply_file(path, stats)
            with h5py.File(path, 'r+') as f:
                store_json(f, 'metadata/inference_config', inference)
        expected = {p.name for p in paths}
        existing = {p.name for p in output_dir.glob('*.hdf5')}
        if existing - expected:
            raise PreprocessError('Output contains other episodes with different statistics; use a fresh output directory')
        manifest = []
        for summary in summaries:
            summary['output'] = str(output_dir / Path(summary['output']).name)
            manifest.append(summary)
        (stage/'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        for name, value in [('env_config.json', env_config), ('normalization.json', stats),
                            ('inference_config.json', inference), ('manifest.json', manifest)]:
            (stage/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths + [stage/name for name in ('config.yaml', 'env_config.json', 'normalization.json', 'inference_config.json', 'manifest.json')]:
            destination = output_dir / path.name
            if cfg['output'].get('overwrite', False):
                os.replace(path, destination)
            else:
                os.link(path, destination)
        for summary in summaries:
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        print(f'Inference bundle: {output_dir / "inference_config.json"}', flush=True)
        return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--input', type=Path, help='Override input file/folder (relative to current directory)')
    parser.add_argument('--output-dir', type=Path, help='Override output directory')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Show crop/first-frame schema plan without writing HDF5')
    args = parser.parse_args()
    try:
        cfg, path = read_settings(args.config)
        if args.overwrite:
            cfg['output']['overwrite'] = True
        summaries = run(cfg, path, args.input, args.output_dir, args.dry_run)
    except (OSError, ValueError, KeyError, SchemaError, yaml.YAMLError) as exc:
        parser.exit(1, f'Preprocess failed: {exc}\n')
    print(f'{len(summaries)} episodes; {sum(s["frames"] for s in summaries)} retained frames; '
          f'{sum(s["skipped"] for s in summaries)} dropped; {"dry run" if args.dry_run else "saved"}')


if __name__ == '__main__':
    main()
