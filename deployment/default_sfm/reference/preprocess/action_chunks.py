"""Hardware-free policy labels matching ChunkInterpreter's world-delta/TCP-relative math."""
import numpy as np
from scipy.spatial.transform import Rotation

from Real_Robo.episode_recording import encode_action_chunk, decode_action_chunk
from .action_layout import MODES, OBSERVED, RESIDUAL, RELATIVE_MODES, arm_width, columns

ABSOLUTE = 'absolute_tcp_xyz_rpy_gripper'


def validate(settings):
    allowed = {'mode', 'length', 'source_format', 'gripper'}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError('actions supports mode, length, source_format, gripper only')
    if 'gripper' in settings:
        from .gripper import settings as gripper_settings
        gripper_settings(settings['gripper'])
    if settings.get('mode', 'delta') not in MODES:
        raise ValueError(f'actions.mode must be one of {MODES}')
    if type(settings.get('length', 50)) is not int or settings.get('length', 50) < 1:
        raise ValueError('actions.length must be a positive integer')
    if settings.get('source_format', 'auto') not in ('auto', 'delta', 'delta_flange', 'absolute'):
        raise ValueError('actions.source_format must be auto, delta, delta_flange or absolute')


class ActionChunks:
    def __init__(self, rows, metadata, collection, decoder, settings, final=None):
        validate(settings)
        self.mode = settings.get('mode', 'delta')
        self.length = settings.get('length', 50)
        self.arm_width = arm_width(self.mode)
        arm = collection.get('train', {}).get('arm_type')
        self.sides = ['left', 'right'] if arm == 'dual' else [arm]
        if any(s not in ('left', 'right') for s in self.sides):
            raise ValueError('Action chunks require train.arm_type left/right/dual')
        source = np.stack([np.asarray(r['actions']) for r in rows])
        if source.ndim != 2 or source.shape[1] != 7 * len(self.sides) or not np.all(np.isfinite(source)):
            raise ValueError('Action chunks require one finite 7D action per arm per timestep')
        hz = float(collection.get('train', {}).get('hz', 10))
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError('Collection train.hz must be positive and finite')
        self.period = 1. / hz
        self.source = source
        self.action_dim = self.arm_width * len(self.sides)
        self.poses = np.empty((len(rows), len(self.sides), 7), dtype=float)
        for side_index, side in enumerate(self.sides):
            pose = np.stack([decoder.channel(r['observations'], side + '/tcp_pose') for r in rows])
            self.poses[:, side_index, :3] = pose[:, :3]
            self.poses[:, side_index, 3:] = Rotation.from_euler('xyz', pose[:, 3:]).as_quat(scalar_first=True)
        declared = {(r.get('infos') or {}).get('action_format') or metadata.get('command_format') or 'unspecified' for r in rows}
        self.source_format = settings.get('source_format', 'auto')
        if self.mode == OBSERVED:
            # Pose supervision needs measured outcomes, not command semantics.
            self.source_format = 'not_used'
        elif self.source_format == 'auto':
            if declared == {ABSOLUTE}:
                self.source_format = 'absolute'
            elif declared <= {'delta', 'world_delta_xyz_rotvec_gripper'}:
                self.source_format = 'delta'
            elif declared <= {'delta_flange', 'flange_delta_xyz_rotvec_gripper'}:
                self.source_format = 'delta_flange'
            else:
                raise ValueError(f'Cannot infer action semantics {declared}; set actions.source_format explicitly')
        self.declared_formats = sorted(declared)
        self.commands = source.astype(float).reshape(len(rows), len(self.sides), 7).copy()
        self.delta = None
        if self.source_format == 'not_used':
            pass
        elif self.source_format == 'absolute':
            self.delta = encode_action_chunk(self.commands, self.poses, None, mode='delta')
        else:
            # A pre-existing world delta is retained verbatim; compose its target at each recorded pose.
            self.commands = decode_action_chunk(source, self.poses, None, mode=self.source_format)
            self.delta = (source.astype(float).copy() if self.source_format == 'delta' else
                          encode_action_chunk(self.commands, self.poses, None, mode='delta'))
        if self.mode == 'delta_flange':
            self.flange_delta = (source.astype(float).copy() if self.source_format == 'delta_flange' else
                                 encode_action_chunk(self.commands, self.poses, None, mode='delta_flange'))
        self.observed_commands = None
        if self.mode in (OBSERVED, RESIDUAL):
            from .convert import next_observation
            following = [next_observation(rows, index, final) for index in range(len(rows))]
            self.observed_commands = source.astype(float).reshape(len(rows), len(self.sides), 7).copy()
            for side_index, side in enumerate(self.sides):
                self.observed_commands[:, side_index, :6] = np.stack([
                    decoder.channel(observation, side + '/tcp_pose') for observation in following])
        self.gripper = None
        if 'gripper' in settings:
            from .gripper import GripperLabels
            self.gripper = GripperLabels(rows, collection, decoder, settings['gripper'], self.sides, final=final)

    def at(self, index):
        count = min(self.length, len(self.source) - index)
        mask = np.arange(self.length) < count
        result = np.zeros((self.length, self.action_dim), dtype=np.float64)
        if self.mode == 'delta':
            result[:count] = self.delta[index:index + count]
        elif self.mode == 'delta_flange':
            result[:count] = self.flange_delta[index:index + count]
        else:
            references = np.broadcast_to(self.poses[index], (count, len(self.sides), 7))
            if self.mode == 'relative':
                result[:count] = encode_action_chunk(self.commands[index:index + count], references, None,
                                                    mode='relative_tcp', gripper_mode='absolute')
            else:
                observed = encode_action_chunk(self.observed_commands[index:index + count], references, None,
                                               mode='relative_tcp', gripper_mode='absolute')
                if self.mode == OBSERVED:
                    result[:count] = observed
                else:
                    command = encode_action_chunk(self.commands[index:index + count], references, None,
                                                  mode='relative_tcp', gripper_mode='absolute')
                    blocks = result[:count].reshape(count, len(self.sides), self.arm_width)
                    observed = observed.reshape(count, len(self.sides), 7)
                    command = command.reshape(count, len(self.sides), 7)
                    blocks[..., :6] = observed[..., :6]
                    blocks[..., 6:12] = command[..., :6] - observed[..., :6]
                    blocks[..., 12] = command[..., 6]
        if self.gripper is not None:
            result[:count, self.arm_width - 1::self.arm_width] = self.gripper.labels[index:index + count]
        return result, mask

    def contract(self):
        result = dict(mode=self.mode, interpreter_mode='relative_tcp' if self.mode in RELATIVE_MODES else self.mode,
                    length=self.length, sides=self.sides, source_format=self.source_format, action_period_s=self.period,
                    declared_formats=self.declared_formats, rotation_unit='rad', translation_unit='m',
                    rotation_representation='rotvec', pose_reference='observations/state at each timestep',
                    arm_action_dim=self.arm_width, columns=columns(self.mode, self.sides),
                    relative_gripper_action_mode='absolute', padding='zero_after_normalization',
                    padding_mask='action_mask', gripper='original command, no relabeling')
        if self.mode in RELATIVE_MODES:
            result.update(action_frame='tcp_local_at_chunk_start',
                          pose_reference='observations/state at chunk start')
        if self.mode in (OBSERVED, RESIDUAL):
            result.update(observed_pose_source='transition next_observations, else same-episode next observations, else final_observation',
                          translation_definition='R[t].T @ (p_next[j] - p[t]); j=t+k',
                          rotation_definition='rotvec(R[t].T @ R_next[j]); j=t+k')
        if self.mode == RESIDUAL:
            result.update(residual_definition='command_relative[:6] - observed_relative[:6] before normalization',
                          execution_definition='observed[:6] + residual[:6], unchanged gripper')
        if self.mode == 'delta_flange':
            result.update(action_frame='flange_local_at_each_step',
                          translation_definition='R[t].T @ (p_target[t] - p_observed[t])',
                          rotation_definition='rotvec(R[t].T @ R_target[t]); right composition',
                          flange_axes='recorded native TCP axes aligned with flange; no tool-origin correction')
        if self.gripper is not None:
            result['gripper'] = self.gripper.contract()
        return result
