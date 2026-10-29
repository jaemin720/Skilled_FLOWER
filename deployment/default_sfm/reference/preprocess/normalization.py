"""Dataset-wide reversible normalization; masked action padding is never fitted."""
import json
from pathlib import Path

import h5py
import numpy as np
from .action_layout import RESIDUAL, arm_width

METHODS = ('none', 'standard', 'minmax', 'symlog', 'symlog_minmax')
ACTION_GROUPS = ('xyz', 'rpy', 'gripper')


def validate_parameters(item, name):
    if item.get('method', 'none') not in METHODS:
        raise ValueError(f'{name}.method must be one of {METHODS}')
    for key, default in [('epsilon', 1e-6), ('temperature', 20.)]:
        value = item.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name}.{key} must be finite and positive')


def group_options(settings, group):
    return {**{k: settings[k] for k in ('method', 'epsilon', 'temperature') if k in settings},
            **settings.get(group, {})}


def action_indices(width, mode=None):
    block = arm_width(mode)
    if width < block or width % block:
        raise ValueError(f'Grouped action normalization requires {block} channels per arm')
    groups = [('xyz', [0, 1, 2]), ('rpy', [3, 4, 5]), ('gripper', [block - 1])]
    if mode == RESIDUAL:
        groups[0][1].extend([6, 7, 8])
        groups[1][1].extend([9, 10, 11])
    return {name: [base + i for base in range(0, width, block) for i in indices]
            for name, indices in groups}


def validate(settings):
    if not isinstance(settings, dict) or set(settings) - {'state', 'action', 'ft', 'stats_file'}:
        raise ValueError('normalization supports state/action/ft/stats_file only')
    if settings.get('stats_file') is not None and not isinstance(settings['stats_file'], str):
        raise ValueError('normalization.stats_file must be a path or null')
    for kind in ('state', 'action', 'ft'):
        item = settings.get(kind, {})
        allowed = {'method', 'epsilon', 'temperature'} | ({'targets'} if kind == 'ft' else set()) | (set(ACTION_GROUPS) if kind == 'action' else set())
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError(f'Invalid normalization.{kind} settings')
        if kind == 'ft':
            targets = item.get('targets', ['ft_history'])
            if (not isinstance(targets, list) or not targets or
                    any(not isinstance(key, str) or not key for key in targets) or len(set(targets)) != len(targets)):
                raise ValueError('normalization.ft.targets must be unique output key names')
        validate_parameters(item, f'normalization.{kind}')
        if kind == 'action':
            for group in ACTION_GROUPS:
                if group in item:
                    value = item[group]
                    if not isinstance(value, dict) or set(value) - {'method', 'epsilon', 'temperature'}:
                        raise ValueError(f'Invalid normalization.action.{group} settings')
                    validate_parameters(group_options(item, group), f'normalization.action.{group}')


class Moments:
    def __init__(self):
        self.count = 0

    def add(self, values):
        x = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        if not len(x):
            return
        if not np.all(np.isfinite(x)):
            raise ValueError('Non-finite normalization input')
        mean = x.mean(axis=0)
        m2 = ((x - mean) ** 2).sum(axis=0)
        low, high = x.min(axis=0), x.max(axis=0)
        if self.count == 0:
            self.mean, self.m2, self.low, self.high = mean, m2, low, high
        else:
            total = self.count + len(x)
            difference = mean - self.mean
            self.m2 += m2 + difference ** 2 * self.count * len(x) / total
            self.mean += difference * len(x) / total
            self.low, self.high = np.minimum(self.low, low), np.maximum(self.high, high)
        self.count += len(x)

    def result(self, config):
        if not self.count:
            raise ValueError('No valid samples for normalization')
        epsilon = float(config.get('epsilon', 1e-6))
        method = config.get('method', 'none')
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.))
        if method == 'standard':
            offset, scale = self.mean, np.maximum(std, epsilon)
        elif method in ('minmax', 'symlog_minmax'):
            offset, scale = (self.low + self.high) / 2, np.maximum((self.high - self.low) / 2, epsilon)
        else:
            offset, scale = np.zeros_like(std), np.ones_like(std)
        return dict(method=method, epsilon=epsilon, temperature=float(config.get('temperature', 20.)),
                    count=self.count, mean=self.mean.tolist(), std=std.tolist(), min=self.low.tolist(), max=self.high.tolist(),
                    offset=offset.tolist(), scale=scale.tolist())


def transform(values, stats, inverse=False):
    x = np.asarray(values, dtype=np.float64)
    if x.shape[-1] != len(stats['offset']) or not np.all(np.isfinite(x)):
        raise ValueError('Normalization input has wrong feature dimension or non-finite values')
    method = stats['method']
    if method == 'grouped':
        result = np.empty_like(x)
        covered = []
        for group in stats['groups'].values():
            indices = group['indices']
            covered.extend(indices)
            result[..., indices] = transform(x[..., indices], group, inverse=inverse)
        if sorted(covered) != list(range(x.shape[-1])):
            raise ValueError('Grouped normalization must cover every action channel exactly once')
    elif method in ('standard', 'minmax'):
        offset, scale = np.asarray(stats['offset']), np.asarray(stats['scale'])
        result = x * scale + offset if inverse else (x - offset) / scale
    elif method in ('symlog', 'symlog_minmax'):
        temperature = stats['temperature']
        if method == 'symlog_minmax':
            offset, scale = np.asarray(stats['offset']), np.asarray(stats['scale'])
            if not inverse:
                x = (x - offset) / scale
        if inverse:
            result = np.sign(x) * np.expm1(np.abs(x) * np.log1p(temperature)) / temperature
            if method == 'symlog_minmax':
                result = result * scale + offset
        else:
            result = np.sign(x) * np.log1p(np.abs(x) * temperature) / np.log1p(temperature)
    elif method == 'none':
        result = x.copy()
    else:
        raise ValueError(f'Unknown normalization method: {method}')
    if not np.all(np.isfinite(result)):
        raise ValueError('Normalization overflow')
    return result


def action_statistics(moments, settings, mode=None):
    result = moments.result(settings)
    if not any(group in settings for group in ACTION_GROUPS):
        return result
    groups = {}
    for name, indices in action_indices(len(result['offset']), mode).items():
        coefficients = moments.result(group_options(settings, name))
        for key in ('mean', 'std', 'min', 'max', 'offset', 'scale'):
            coefficients[key] = [coefficients[key][i] for i in indices]
        for key in ('offset', 'scale'):
            for index, value in zip(indices, coefficients[key]):
                result[key][index] = value
        groups[name] = dict(indices=indices, **coefficients)
    return dict(result, method='grouped', groups=groups)


def check_parameters(actual, expected):
    for key, default in [('method', 'none'), ('epsilon', 1e-6), ('temperature', 20.)]:
        if actual[key] != expected.get(key, default):
            raise ValueError(f'Saved normalization {key} differs from config')


def check_action_statistics(actual, settings, width, mode=None):
    grouped = any(group in settings for group in ACTION_GROUPS)
    if grouped != (actual['method'] == 'grouped'):
        raise ValueError('Saved action normalization grouping differs from config')
    if not grouped:
        check_parameters(actual, settings)
        return
    expected_indices = action_indices(width, mode)
    if set(actual['groups']) != set(ACTION_GROUPS):
        raise ValueError('Saved action normalization groups differ from config')
    for name, indices in expected_indices.items():
        if actual['groups'][name]['indices'] != indices:
            raise ValueError('Saved action normalization channel indices differ from config')
        check_parameters(actual['groups'][name], group_options(settings, name))


def schema(path, cfg):
    with h5py.File(path) as f:
        states = {e['target']: json.loads(f['observations/' + e['target']].attrs['columns'])
                  for e in cfg['observations'] if e['type'] == 'state'}
        action = json.loads(f['metadata/action_contract'][()]) if 'metadata/action_contract' in f else {'mode': 'raw'}
        # Declaration wording may differ, while the explicit effective convention must agree.
        action.pop('declared_formats', None)
        return dict(state_columns=states, observations=cfg['observations'],
                    shapes={key: list(value.shape[1:]) for key, value in f['observations'].items()},
                    action_shape=list(f['actions'].shape[1:]), action=action)


def observation_settings(signature, settings):
    """Choose state outputs and explicitly selected F/T array outputs."""
    selected = {key: settings.get('state', {}) for key in signature['state_columns']}
    if 'ft' in settings:
        ft = settings['ft']
        entries = {e['target']: e for e in signature['observations']}
        for key in ft.get('targets', ['ft_history']):
            shape = signature['shapes'].get(key, [])
            if key not in entries or entries[key]['type'] != 'array' or not shape or shape[-1] != 6:
                raise ValueError(f'normalization.ft target {key!r} must be an array with last dimension 6')
            selected[key] = ft
    return selected


def fit_or_load(paths, cfg, stats_file=None, progress=None):
    if progress:
        progress(stage='validate', message='출력 스키마 및 action 규약 확인',
                 current=0, total=len(paths), unit='files', force=True)
    signature = schema(paths[0], cfg)
    if progress:
        progress(stage='validate', message='출력 스키마 및 action 규약 확인',
                 current=1, total=len(paths), unit='files')
    for index, path in enumerate(paths[1:], 2):
        if schema(path, cfg) != signature:
            raise ValueError('Episode output schema/convention differs; split into separate preprocessing runs')
        if progress:
            progress(stage='validate', message='출력 스키마 및 action 규약 확인',
                     current=index, total=len(paths), unit='files', file_path=str(path))
    settings = cfg.get('normalization', {})
    selected = observation_settings(signature, settings)
    if stats_file is not None:
        if progress:
            progress(stage='statistics', message='기존 정규화 통계 읽기 및 검증',
                     file_path=str(stats_file), force=True)
        result = json.loads(Path(stats_file).read_text())
        if result.get('schema') != signature:
            raise ValueError('Saved normalization schema does not match selected state/chunk settings')
        if set(result['observations']) != set(selected):
            raise ValueError('Saved normalization observation targets differ (including F/T)')
        pairs = [(result['observations'][key], options) for key, options in selected.items()]
        for actual, expected in pairs:
            check_parameters(actual, expected)
        check_action_statistics(result['actions'], settings.get('action', {}), signature['action_shape'][-1],
                                signature['action']['mode'])
        result['reused_from'] = str(Path(stats_file).resolve())
        if progress:
            progress(stage='statistics', message='기존 정규화 통계 재사용',
                     current=1, total=1, unit='files', file_path=str(stats_file), force=True)
        return result
    state_moments = {key: Moments() for key in selected}
    action_moments = Moments()
    for file_index, path in enumerate(paths, 1):
        with h5py.File(path) as f:
            count = len(f['actions'])
            context = dict(file_path=str(path), file_index=file_index, file_total=len(paths))
            if progress:
                progress(stage='statistics', message='정규화 통계 계산', current=0,
                         total=count, unit='frames', force=True, **context)
            for start in range(0, count, 128):
                stop = min(start + 128, count)
                for key, moments in state_moments.items():
                    moments.add(f['observations/' + key][start:stop])
                actions = f['actions'][start:stop]
                if 'action_mask' in f:
                    actions = actions[f['action_mask'][start:stop]]
                action_moments.add(actions)
                if progress:
                    progress(stage='statistics', message='정규화 통계 계산', current=stop,
                             total=count, unit='frames', **context)
    return dict(version=1, schema=signature, fit_scope='retained training observations (F/T flattened across timestep and history samples); valid action chunk entries',
                observations={key: m.result(selected[key]) for key, m in state_moments.items()},
                actions=action_statistics(action_moments, settings.get('action', {}), signature['action']['mode']))


def apply_file(path, stats, progress=None):
    with h5py.File(path, 'r+') as f:
        count = len(f['actions'])
        if progress:
            progress(stage='normalize', message='정규화 적용', current=0, total=count,
                     unit='frames', file_path=str(path), force=True)
        for start in range(0, count, 128):
            stop = min(start + 128, count)
            for key, coefficients in stats['observations'].items():
                for group in ('observations', 'next_observations'):
                    if group in f:
                        ds = f[group + '/' + key]
                        ds[start:stop] = transform(ds[start:stop], coefficients)
            values = transform(f['actions'][start:stop], stats['actions'])
            if 'action_mask' in f:
                values[~f['action_mask'][start:stop]] = 0.
            f['actions'][start:stop] = values
            if progress:
                progress(stage='normalize', message='정규화 적용', current=stop,
                         total=count, unit='frames', file_path=str(path))
        f.create_dataset('metadata/normalization', data=json.dumps(stats, ensure_ascii=False), dtype=h5py.string_dtype('utf-8'))
        f.attrs['normalization'] = 'metadata/normalization'
        f['actions'].attrs['normalization'] = stats['actions']['method']
        for key, coefficients in stats['observations'].items():
            for group in ('observations', 'next_observations'):
                if group in f:
                    f[group + '/' + key].attrs['normalization'] = coefficients['method']
