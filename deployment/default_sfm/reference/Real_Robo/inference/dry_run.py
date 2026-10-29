"""Exercise the full normalized RPC and HTML journal using an HDF5 episode only."""
import time
import h5py
import numpy as np
from .protocol import bundle_id, validate_actions, validate_server
from .monitor import EpisodeJournal, MonitorServer, virtual_paths
from preprocess.inference import InferencePreprocessor


def run_bundle_dry_run(args, bundle, client):
    recorded = InferencePreprocessor(args.dry_run_hdf5)
    if bundle_id(recorded) != bundle_id(bundle):
        raise ValueError('Dry-run HDF5 must use the same bundle')
    health = client.health()
    validate_server(bundle, {**health, 'simulation_only': False})
    journal = EpisodeJournal(args.record_dir)
    monitor = None
    try:
        if not args.no_monitor:
            monitor = MonitorServer(journal, args.monitor_port)
            print(f'Offline HTML monitor: http://127.0.0.1:{monitor.port}', flush=True)
            if not getattr(args, 'no_browser', False):
                monitor.open_browser()
        journal.begin(dict(inference_config=bundle.config, normalization=bundle.stats,
                           policy_sides=bundle.config['actions']['sides'],
                           mode=bundle.config['actions']['interpreter_mode'],
                           simulation_only=True, coordinate_frame='cell-base (m), WXYZ',
                           observation_source='already preprocessed HDF5; no second normalization'))
        client.reset(args.prompt)
        sides = bundle.config['actions']['sides']
        paused = getattr(args, 'start_paused', False)
        outcome = 'unknown'
        with h5py.File(args.dry_run_hdf5) as f:
            count = min(len(f['actions']), args.dry_run_queries)
            t = 0
            while t < count:
                finish = False
                for command in journal.pop_controls():
                    if command in ('success','failure','skip'):
                        outcome, finish = command, True
                    elif command == 'pause':
                        paused = True
                    elif command == 'resume':
                        paused = False
                if finish:
                    break
                if paused:
                    journal.set_status('paused')
                    time.sleep(.1)
                    continue
                observation = {k: f['observations'][k][t] for k in f['observations']}
                poses = dict(zip(sides, f['action_reference_poses'][t]))
                journal.frame(t, observation, poses, phase='offline')
                rid = journal.prediction(journal.active, t, observation, poses)
                response = client.infer(observation)
                normalized = validate_actions(bundle, response['actions'])
                physical = bundle.denormalize_actions(normalized)
                physical = bundle.compose_actions(physical)
                execution = physical
                if bundle.uses_opening_gripper:
                    from rb_env.gripper_interface import GripperInterface
                    interfaces = bundle.config['actions']['gripper']['interfaces']
                    observed = [float(GripperInterface.from_contract(interfaces[side]).observed_opening(f['gripper_observed'][t, i]))
                                for i, side in enumerate(sides)]
                    execution = bundle.decode_gripper_actions(physical, observed_gripper=observed,
                        previous_openings=f['gripper_previous_commands'][t])
                contract = bundle.config['actions']
                paths = virtual_paths(execution, poses, sides, contract['interpreter_mode'],
                                      contract.get('relative_action_frame_yaw_deg', 0.))
                journal.response(rid, normalized, physical, paths)
                t += 1
                if monitor:
                    time.sleep(.1)
        journal.finish(outcome)
        journal.flush()
        print(f"Offline RPC passed: {t} requests, {journal.root if journal.root else 'disk recording disabled'}", flush=True)
    finally:
        try:
            if monitor:
                monitor.close()
        finally:
            journal.close()
