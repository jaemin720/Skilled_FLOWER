"""Episode-local gripper labels and an explicit adapter back to absolute commands."""
import copy
from collections import deque

import numpy as np
from rb_env.gripper_interface import GripperInterface


LEGACY_DEFAULTS = dict(mode='absolute', source_mode='auto', offset=.02, deadband=.001,
                command_range=[-1., 1.], observation_range=[0., 1.],
                grasp=dict(enabled=True, stationary_threshold=.005, closed_threshold=.05,
                           command_gap=.02, consecutive_frames=3))

DEFAULTS = dict(label_version=2, mode='absolute', offset=.02, deadband=.001,
                grasp=dict(enabled=True, stationary_tolerance=.005, stationary_duration_s=.3))


def settings(value):
    if not isinstance(value, dict):
        raise ValueError('Invalid actions.gripper settings')
    version = value.get('label_version')
    if version is not None and (type(version) is not int or version not in (1, 2)):
        raise ValueError('label_version must be 1 or 2')
    # Even a minimal old YAML (mode/offset only) must retain its command units.
    # The UI always writes the explicit version when enabling or migrating labels.
    if version in (None, 1):
        return dict(legacy_settings({k: v for k, v in value.items() if k != 'label_version'}), label_version=1)
    if version not in (None, 2) or set(value) - (set(DEFAULTS) | {'legacy'}):
        raise ValueError('Invalid actions.gripper settings or label_version')
    result = copy.deepcopy(DEFAULTS)
    result.update({k: v for k, v in value.items() if k != 'grasp'})
    if 'grasp' in value:
        if not isinstance(value['grasp'], dict) or set(value['grasp']) - set(DEFAULTS['grasp']):
            raise ValueError('Invalid observation-based actions.gripper.grasp settings')
        result['grasp'].update(value['grasp'])
    if result['mode'] not in ('absolute', 'relative', 'binary'):
        raise ValueError('actions.gripper.mode must be absolute/relative/binary')
    for key in ('offset', 'deadband'):
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f'actions.gripper.{key} must be an opening fraction in [0,1]')
    g = result['grasp']
    if not isinstance(g['enabled'], bool):
        raise ValueError('actions.gripper.grasp.enabled must be true/false')
    for key in ('stationary_tolerance', 'stationary_duration_s'):
        v = g[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
            raise ValueError(f'actions.gripper.grasp.{key} must be positive and finite')
    if g['stationary_tolerance'] >= 1:
        raise ValueError('stationary_tolerance must be less than one full opening')
    legacy = result.get('legacy', {})
    if not isinstance(legacy, dict) or set(legacy) - {'source_mode', 'observation_range', 'command_range', 'input_range', 'output_range'}:
        raise ValueError('Invalid legacy gripper unit overrides')
    if legacy.get('source_mode', 'auto') not in ('auto', 'absolute', 'binary'):
        raise ValueError('Invalid legacy.source_mode')
    for key in ('observation_range', 'command_range', 'input_range', 'output_range'):
        if key in legacy:
            pair = legacy[key]
            if not isinstance(pair, list) or len(pair) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in pair) or pair[0] == pair[1]:
                raise ValueError(f'Invalid legacy.{key}')
    return result


def observed_holds(values, times, valid, tolerance, duration):
    """Close/stop/open segments using a rolling confirmation window, without a hold limit."""
    mask = np.zeros(len(values), dtype=bool)
    low, high = deque(), deque()
    start, armed, holding = 0, False, False
    for t, value in enumerate(values):
        if not valid[t]:
            low.clear(); high.clear()
            start, armed, holding = t + 1, False, False
            continue
        # Keep one boundary sample so irregular timestamps still span `duration`.
        # Older extrema must not erase an entire long hold when opening resumes.
        while start < t and times[t] - times[start + 1] >= duration - 1e-9:
            start += 1
        while low and low[0] < start: low.popleft()
        while high and high[0] < start: high.popleft()
        previous_low = low[0] if low else None
        previous_high = high[0] if high else None
        opening = previous_low is not None and value > values[previous_low] + tolerance
        closing = previous_high is not None and value < values[previous_high] - tolerance
        if opening or closing:
            if holding:
                pivot = previous_low if opening else previous_high
                mask[pivot + 1:t + 1] = False
            holding = False
            armed = bool(closing)
        while low and values[low[-1]] >= value:
            low.pop()
        while high and values[high[-1]] <= value:
            high.pop()
        low.append(t); high.append(t)
        while values[high[0]] - values[low[0]] > tolerance:
            start += 1
            while low[0] < start: low.popleft()
            while high[0] < start: high.popleft()
        if armed and times[t] - times[start] >= duration - 1e-9:
            if not holding:
                mask[start:t + 1] = True
                holding = True
            else:
                mask[t] = True
    return mask


def observation_receipt(row, side, count):
    info = row.get('infos') or {}
    return (info.get('command_receipts') or {}).get(side, info if count == 1 else {})


class GripperLabels:
    def __init__(self, rows, collection, decoder, options, sides, final=None):
        cfg = settings(options)
        if cfg['label_version'] == 1:
            self._legacy = LegacyGripperLabels(rows, collection, decoder,
                {k: v for k, v in cfg.items() if k != 'label_version'}, sides, final)
            self.__dict__.update(self._legacy.__dict__)
            return
        self.settings, self.sides = cfg, sides
        self.interfaces = [GripperInterface.from_collection(collection, side, cfg.get('legacy')) for side in sides]
        self.source_modes = ['binary' if interface.control.mode == 'binary' else 'absolute' for interface in self.interfaces]
        self.observed = np.stack([[float(decoder.channel(r['observations'], s + '/gripper_pose')[0]) for s in sides] for r in rows])
        self.opening = np.column_stack([interface.observed_opening(self.observed[:, i]) for i, interface in enumerate(self.interfaces)])
        self.observation_valid = np.ones(self.opening.shape, dtype=bool)
        for t, row in enumerate(rows):
            for i, (side, interface) in enumerate(zip(sides, self.interfaces)):
                sample = (row.get('action_context') or {}).get('gripper_observations', {}).get(side)
                if sample is None:
                    prior = rows[t - 1] if t else row
                    sample = observation_receipt(prior, side, len(sides)).get('gripper_observation')
                if sample is not None:
                    self.observation_valid[t, i] = interface.observation(sample)['valid']
        self.next_observed = None
        next_opening = None
        if cfg['mode'] != 'binary':
            from .convert import next_observation
            self.next_observed = np.stack([[float(decoder.channel(next_observation(rows, t, final), s + '/gripper_pose')[0])
                                           for s in sides] for t in range(len(rows))])
            next_opening = np.column_stack([interface.observed_opening(self.next_observed[:, i]) for i, interface in enumerate(self.interfaces)])
        self.commands = np.empty_like(self.opening)
        for i, (side, interface) in enumerate(zip(sides, self.interfaces)):
            previous = self.opening[0, i]
            for t, row in enumerate(rows):
                receipt = observation_receipt(row, side, len(sides))
                percent = receipt.get('gripper_target_percent')
                if percent is not None:
                    if not np.isfinite(percent): raise ValueError('gripper_target_percent must be finite')
                    previous = float(np.clip(percent / 100., 0., 1.))
                else:
                    previous = interface.target_from_command(np.asarray(row['actions'])[6 + i * 7], previous)
                self.commands[t, i] = previous
        self.previous = np.vstack((self.commands[:1], self.commands[:-1]))
        self.difference = self.commands - self.previous
        hz = float(collection.get('train', {}).get('hz', 10))
        if not np.isfinite(hz) or hz <= 0: raise ValueError('Collection train.hz must be positive and finite')
        times = []
        for t, row in enumerate(rows):
            value = row.get('time', (row.get('action_context') or {}).get('timestamp', t / hz))
            try: times.append(float(value))
            except (ValueError, TypeError): times.append(float('nan'))
        times = np.asarray(times)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            times = np.arange(len(rows)) / hz
        self.grasp = np.zeros_like(self.opening, dtype=bool)
        g = cfg['grasp']
        if g['enabled']:
            for i in range(len(sides)):
                self.grasp[:, i] = observed_holds(self.opening[:, i], times, self.observation_valid[:, i],
                                                 g['stationary_tolerance'], g['stationary_duration_s'])
        if cfg['mode'] == 'absolute':
            # Do not add a closing bias to a target already opening next step.
            bias = self.grasp & (next_opening <= self.opening + g['stationary_tolerance'])
            self.labels = next_opening - cfg['offset'] * bias
        elif cfg['mode'] == 'relative':
            difference = next_opening - self.opening
            moving = np.abs(difference) > cfg['deadband']
            self.labels = np.where(moving, difference, np.where(self.grasp, -cfg['offset'], 0.))
        else:
            self.labels = np.where(np.abs(self.difference) > cfg['deadband'], np.sign(self.difference), 0.)

    def contract(self):
        if hasattr(self, '_legacy'):
            return self._legacy.contract()
        return dict(**self.settings, label_unit='opening_fraction',
                    sides=list(self.sides),
                    label_source='command' if self.settings['mode'] == 'binary' else 'next_state',
                    interfaces={side: interface.contract() for side, interface in zip(self.sides, self.interfaces)},
                    source_modes=self.source_modes, grasp_source='observed closing then stationary window until opening; full closure included',
                    grasp_window='rolling stationary_duration_s window with one boundary sample; no maximum hold duration',
                    offset_direction='subtract positive opening fraction; absolute opening transitions have no bias',
                    validity='invalid/stale measured samples reset grasp; old recordings without status assume valid',
                    adapter='decode_gripper_actions returns opening targets; map through live GripperInterface before execution')


def legacy_settings(value):
    if not isinstance(value, dict) or set(value) - set(LEGACY_DEFAULTS):
        raise ValueError('Invalid actions.gripper settings')
    result = copy.deepcopy(LEGACY_DEFAULTS)
    result.update({k: v for k, v in value.items() if k != 'grasp'})
    if 'grasp' in value:
        if not isinstance(value['grasp'], dict) or set(value['grasp']) - set(LEGACY_DEFAULTS['grasp']):
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


def observation_commands(values, cfg):
    """Map measured openings to the command units used by labels and offsets."""
    obs_lo, obs_hi = cfg['observation_range']
    opening = (np.asarray(values, dtype=float) - obs_lo) / (obs_hi - obs_lo)
    if not np.all(np.isfinite(opening)) or np.any(opening < -1e-6) or np.any(opening > 1 + 1e-6):
        raise ValueError('Gripper observations exceed actions.gripper.observation_range; check units and state layout')
    lo, hi = cfg['command_range']
    return lo + opening * (hi - lo)


class LegacyGripperLabels:
    def __init__(self, rows, collection, decoder, options, sides, final=None):
        self.settings = cfg = legacy_settings(options)
        self.sides = sides
        raw = np.stack([np.asarray(r['actions'], dtype=float)[6::7] for r in rows])
        self.observed = np.stack([[float(decoder.channel(row['observations'], side + '/gripper_pose')[0])
                                   for side in sides] for row in rows])
        lo, hi = cfg['command_range']
        observed_commands = observation_commands(self.observed, cfg)
        opening = (observed_commands - lo) / (hi - lo)
        self.next_observed = None
        if cfg['mode'] != 'binary':
            from .convert import next_observation
            self.next_observed = np.stack([
                [float(decoder.channel(next_observation(rows, t, final), side + '/gripper_pose')[0])
                 for side in sides] for t in range(len(rows))])
            next_commands = observation_commands(self.next_observed, cfg)
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
        if cfg['mode'] == 'absolute':
            self.labels = next_commands - cfg['offset'] * self.grasp
        elif cfg['mode'] == 'relative':
            state_difference = next_commands - observed_commands
            moving = np.abs(state_difference) > cfg['deadband']
            self.labels = np.where(moving, state_difference, 0.)
            self.labels = np.where(self.grasp & ~moving, -cfg['offset'], self.labels)
        else:
            moving = np.abs(difference) > cfg['deadband']
            self.labels = np.where(moving, np.sign(difference), 0.)

    def contract(self):
        return dict(**self.settings, source_modes=self.source_modes,
                    label_source='command' if self.settings['mode'] == 'binary' else 'next_state',
                    state_units='observation_range linearly mapped to command_range',
                    difference_reference=('current absolute target minus previous target in same episode; first difference=0'
                                          if self.settings['mode'] == 'binary' else
                                          'next observed state minus current observed state, in command units'),
                    next_state_source='transition next_observations, else same-episode next observations, else final_observation',
                    offset_direction='subtract positive offset to close; relative offset only replaces zero',
                    binary='target increased=+1, decreased=-1, unchanged=0; no grasp offset',
                    adapter='decode_gripper_actions for relative/binary before ChunkInterpreter')


def decode(values, contract, previous_commands=None, *, observed_gripper=None, previous_openings=None):
    """Decode state-relative labels from measured openings; legacy/binary use commands."""
    x = np.asarray(values, dtype=float).copy()
    if x.ndim != 2 or x.shape[1] % 7 or not np.all(np.isfinite(x)):
        raise ValueError('Actions must be finite [H, 7*sides]')
    mode = contract['mode']
    if mode == 'absolute':
        return x
    if contract.get('label_version') == 2:
        interfaces = [GripperInterface.from_contract(contract['interfaces'][side]) for side in contract.get('sides', contract['interfaces'])]
        if mode == 'relative':
            if observed_gripper is None:
                raise ValueError('State-relative grippers require observed_gripper at prediction time')
            observed = np.asarray(observed_gripper, dtype=float)
            if observed.shape != (len(interfaces),):
                raise ValueError('observed_gripper must contain one value per arm')
            if not np.isfinite(observed).all() or np.any(observed < 0) or np.any(observed > 1):
                raise ValueError('observed_gripper must be canonical opening fractions in [0,1]')
            start = observed
            x[:, 6::7] = start + np.cumsum(x[:, 6::7], axis=0)
        elif mode == 'binary':
            start = np.asarray(previous_openings, dtype=float)
            if start.shape != (len(interfaces),) or not np.isfinite(start).all():
                raise ValueError('Binary grippers require previous_openings: one last target opening per arm')
            targets = np.asarray([interface.binary_targets() for interface in interfaces])
            for t in range(len(x)):
                start = np.where(x[t, 6::7] >= .5, targets[:, 1], np.where(x[t, 6::7] <= -.5, targets[:, 0], start))
                x[t, 6::7] = start
        else:
            raise ValueError('Unsupported gripper label mode')
        return x
    if mode == 'relative' and contract.get('label_source') == 'next_state':
        if observed_gripper is None:
            raise ValueError('next_state relative grippers require observed_gripper at prediction time, not previous commands')
        start = observation_commands(observed_gripper, contract)
        reference = 'observed_gripper'
    else:
        start = np.asarray(previous_commands, dtype=float)
        reference = 'previous_commands'
    if start.shape != (x.shape[1] // 7,) or not np.all(np.isfinite(start)):
        raise ValueError(f'{reference} must contain one finite value per arm')
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
