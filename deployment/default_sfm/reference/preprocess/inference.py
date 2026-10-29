"""Reuse recorded preprocessing and normalization without fitting on inference inputs."""
import json
import copy
from pathlib import Path

import h5py
import numpy as np

from .convert import StateDecoder, transform_observation
from .normalization import transform
from .action_layout import execution_labels


class InferencePreprocessor:
    def __init__(self, bundle):
        path = Path(bundle)
        if path.is_file() and path.suffix in ('.h5', '.hdf5'):
            with h5py.File(path) as f:
                self.config = json.loads(f['metadata/inference_config'][()])
                self.stats = json.loads(f['metadata/normalization'][()])
        else:
            self.config = json.loads((path/'inference_config.json').read_text())
            self.stats = json.loads((path/self.config['normalization_file']).read_text())
        self.decoder = StateDecoder(self.config['collection_config'])
        expected_columns = {e['target']: self.decoder.columns(e['keys'])
                            for e in self.config['observations'] if e['type'] == 'state'}
        action = dict(self.config['actions'])
        action.pop('declared_formats', None)
        signature = self.stats.get('schema', {})
        if (self.config.get('version') != 1 or self.stats.get('version') != 1 or
                signature.get('observations') != self.config['observations'] or
                signature.get('state_columns') != expected_columns or signature.get('action') != action):
            raise ValueError('Inference config and normalization statistics do not match')

    def observation(self, observation, collection_config=None):
        decoder = StateDecoder(collection_config) if collection_config else self.decoder
        result = transform_observation(observation, self.config['observations'], decoder)
        for key, coefficients in self.stats['observations'].items():
            result[key] = transform(result[key], coefficients).astype(np.float32)
        return result

    def normalize_actions(self, values):
        return transform(values, self.stats['actions'])

    @property
    def uses_opening_gripper(self):
        gripper = self.config['actions'].get('gripper')
        return isinstance(gripper, dict) and gripper.get('label_version') == 2

    def execution_config(self, config):
        """Select continuous native targets for the new canonical policy adapter only."""
        if not self.uses_opening_gripper:
            return config
        from rb_env.gripper_interface import GripperInterface
        from rb_env.gripper_drivers.base import GripperControlConfig
        config = copy.deepcopy(config)
        for side in self.config['actions']['sides']:
            interface = GripperInterface.from_collection(config, side)
            control = dict(interface.contract()['control'], mode='continuous')
            config.setdefault('env', {}).setdefault(side, {}).setdefault('gripper_control', {}).update(control)
            config.setdefault('gripper_interfaces', {})[side] = GripperInterface(
                GripperControlConfig().with_overrides(control), interface.stroke_m).contract()
        return config

    def denormalize_actions(self, values):
        """Restore stored labels; call compose_actions before decoding grippers or execution."""
        return transform(values, self.stats['actions'], inverse=True)

    def compose_actions(self, values):
        """Convert denormalized stored labels to 7 channels per arm for execution."""
        return execution_labels(values, self.config['actions']['mode'])

    def decode_gripper_actions(self, values, previous_commands=None, *, observed_gripper=None, previous_openings=None):
        """Use raw measured openings for state-relative, last targets for binary/legacy."""
        gripper = self.config['actions'].get('gripper')
        if not isinstance(gripper, dict):
            return np.asarray(values, dtype=float).copy()
        from .gripper import decode
        return decode(values, gripper, previous_commands, observed_gripper=observed_gripper, previous_openings=previous_openings)

    def interpreter_options(self, *, decoded_gripper=False):
        action = self.config['actions']
        if action['mode'] == 'raw':
            raise ValueError('This legacy bundle has no policy chunk convention')
        gripper = action.get('gripper')
        if isinstance(gripper, dict) and (gripper['mode'] != 'absolute' or self.uses_opening_gripper) and not decoded_gripper:
            raise ValueError('Use decode_gripper_actions with observed_gripper for next_state relative labels '
                             'or previous_openings for canonical binary labels (previous_commands for legacy), '
                             'map canonical targets through PolicyIO.execution_actions, then interpreter_options(decoded_gripper=True)')
        return dict(mode=action['interpreter_mode'], policy_sides=tuple(action['sides']),
                    relative_gripper_action_mode=action['relative_gripper_action_mode'],
                    relative_action_frame_yaw_deg=action.get('relative_action_frame_yaw_deg', 0.),
                    action_period_s=self.config['action_period_s'])
