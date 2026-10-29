"""Episode-local gripper labels and an explicit adapter back to absolute commands."""
import copy

import numpy as np


DEFAULTS = dict(mode='absolute', source_mode='auto', offset=.02, deadband=.001,
                command_range=[-1., 1.], observation_range=[0., 1.],
                grasp=dict(enabled=True, stationary_threshold=.005, closed_threshold=.05,
                           command_gap=.02, consecutive_frames=3))


def settings(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('Invalid actions.gripper settings')
    result = copy.deepcopy(DEFAULTS)
    result.update({k: v for k, v in value.items() if k != 'grasp'})
    if 'grasp' in value:
        if not isinstance(value['grasp'], dict) or set(value['grasp']) - set(DEFAULTS['grasp']):
            raise ValueError('Invalid actions.gripper.grasp settings')
        result['grasp'].update(value['grasp'])
    if result['mode'] not in ('absolute', 'relative', 'binary'):
        raise ValueError('actions.gripper.mode must be absolute/relative/binary')
    if result['source_mode'] not in ('auto', 'absolute', 'binary'):
        raise ValueError('actions.gripper.source_mode must be auto/absolute/binary')
    for key in ('command_range', 'observation_range'):
        pair = result[key]
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in pair)
                or pair[0] >= pair[1]):
            raise ValueError(f'actions.gripper.{key} must be [closed, open] with closed < open')
    for name, section, keys in [('gripper', result, ('offset', 'deadband')),
                               ('gripper.grasp', result['grasp'], ('stationary_threshold', 'closed_threshold', 'command_gap'))]:
        for key in keys:
            v = section[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f'actions.{name}.{key} must be finite and nonnegative')
    grasp = result['grasp']
    if not isinstance(grasp['enabled'], bool):
        raise ValueError('actions.gripper.grasp.enabled must be true/false')
    if grasp['closed_threshold'] >= 1:
        raise ValueError('actions.gripper.grasp.closed_threshold must be < 1 (opening fraction)')
    if type(grasp['consecutive_frames']) is not int or grasp['consecutive_frames'] < 1:
        raise ValueError('actions.gripper.grasp.consecutive_frames must be a positive integer')
    return result


class GripperLabels:
    def __init__(self, rows, collection, decoder, options, sides):
        self.settings = cfg = settings(options)
        self.sides = sides
        raw = np.stack([np.asarray(r['actions'], dtype=float)[6::7] for r in rows])
        self.observed = np.stack([[float(decoder.channel(row['observations'], side + '/gripper_pose')[0])
                                   for side in sides] for row in rows])
        obs_lo, obs_hi = cfg['observation_range']
        opening = (self.observed - obs_lo) / (obs_hi - obs_lo)
        if not np.all(np.isfinite(opening)) or np.any(opening < -1e-6) or np.any(opening > 1 + 1e-6):
            raise ValueError('Gripper observations exceed actions.gripper.observation_range; check units and state layout')
        lo, hi = cfg['command_range']
        observed_commands = lo + opening * (hi - lo)
        self.commands = raw.copy()
        self.source_modes = []
        for s, side in enumerate(sides):
            control = collection.get('env', {}).get(side, {}).get('gripper_control', {})
            mode = cfg['source_mode']
            if mode == 'auto':
                mode = 'binary' if control.get('mode') == 'binary' else 'absolute'
            self.source_modes.append(mode)
            if mode != 'binary':
                continue
            close = control.get('binary_close_threshold_m')
            open_ = control.get('binary_open_threshold_m')
            if close is None:
                close, open_ = -.5, .5
            if open_ is None or not np.isfinite(close) or not np.isfinite(open_) or close >= open_:
                raise ValueError(f'{side}: invalid source binary gripper thresholds')
            previous = observed_commands[0, s]
            for t, row in enumerate(rows):
                info = row.get('infos') or {}
                receipt = (info.get('command_receipts') or {}).get(side, info if len(sides) == 1 else {})
                percent = receipt.get('gripper_target_percent')
                if percent is not None:
                    if not np.isfinite(percent) or not 0 <= percent <= 100:
                        raise ValueError('gripper_target_percent must be finite and within [0,100]')
                    previous = lo + float(percent) / 100 * (hi - lo)
                elif control.get('binary_closed_width_m') is not None:
                    raise ValueError('Binary custom-width recordings need gripper_target_percent receipts to reconstruct targets')
                elif np.float32(raw[t, s]) <= np.float32(close):
                    previous = lo
                elif np.float32(raw[t, s]) >= np.float32(open_):
                    previous = hi
                self.commands[t, s] = previous
        self.previous = np.vstack((self.commands[:1], self.commands[:-1]))
        difference = self.commands - self.previous
        self.difference = difference.copy()
        self.grasp = np.zeros(raw.shape, dtype=bool)
        g = cfg['grasp']
        if g['enabled']:
            stationary = np.zeros_like(self.grasp)
            stationary[1:] = np.abs(np.diff(opening, axis=0)) <= g['stationary_threshold']
            candidates = (stationary & (opening > g['closed_threshold']) &
                          (self.commands < observed_commands - g['command_gap']) &
                          (difference <= cfg['deadband']))
            run = np.zeros(len(sides), dtype=int)
            for t in range(len(rows)):
                run = np.where(candidates[t], run + 1, 0)
                self.grasp[t] = run >= g['consecutive_frames']
        moving = np.abs(difference) > cfg['deadband']
        if cfg['mode'] == 'absolute':
            self.labels = self.commands - cfg['offset'] * self.grasp
        elif cfg['mode'] == 'relative':
            self.labels = np.where(moving, difference, 0.)
            self.labels = np.where(self.grasp & ~moving, -cfg['offset'], self.labels)
        else:
            self.labels = np.where(moving, np.sign(difference), 0.)

    def contract(self):
        return dict(**self.settings, source_modes=self.source_modes,
                    difference_reference='current absolute target minus previous target in same episode; first difference=0',
                    offset_direction='subtract positive offset to close; relative offset only replaces zero',
                    binary='target increased=+1, decreased=-1, unchanged=0; no grasp offset',
                    adapter='decode_gripper_actions for relative/binary before ChunkInterpreter')


def decode(values, contract, previous_commands):
    """Convert label-space grippers to absolute targets; caller supplies last command."""
    x = np.asarray(values, dtype=float).copy()
    if x.ndim != 2 or x.shape[1] % 7 or not np.all(np.isfinite(x)):
        raise ValueError('Actions must be finite [H, 7*sides]')
    start = np.asarray(previous_commands, dtype=float)
    if start.shape != (x.shape[1] // 7,) or not np.all(np.isfinite(start)):
        raise ValueError('previous_commands must contain one last absolute gripper command per arm')
    mode = contract['mode']
    if mode == 'relative':
        x[:, 6::7] = start + np.cumsum(x[:, 6::7], axis=0)
    elif mode == 'binary':
        lo, hi = contract['command_range']
        for t in range(len(x)):
            start = np.where(x[t, 6::7] >= .5, hi, np.where(x[t, 6::7] <= -.5, lo, start))
            x[t, 6::7] = start
    elif mode != 'absolute':
        raise ValueError('Unsupported gripper label mode')
    return x
