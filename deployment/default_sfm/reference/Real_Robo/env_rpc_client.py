if __name__ == "__main__":
    from Real_Robo.ros_environment import ensure_ros_environment

    ensure_ros_environment()

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rb_env.envs.builder_integration import (
    build_environment,
    load_config,
    resolve_action_chunk_mode,
    resolve_policy_sides,
    resolve_send_chunk_length,
)
from rb_env.envs.chunk_interpreter import ChunkInterpreter
from .resource_paths import config_dir


DEFAULT_ENV_CONFIG_DIR = config_dir()

# Re-export helpers for existing callers; implementations live in inference/.
from .inference.client import HTTPPolicyClient
from .inference.policy_io import (POLICY_BLOCK_DIM, PolicyIO, response_to_chunk,
                                  episode_result,
                                  resolve_server_contract, compensate_chunk_delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer_ip", type=str, default="115.145.179.169")
    parser.add_argument("--trainer_port", type=int, default=45587)
    parser.add_argument(
        "--env_path",
        type=str,
        default=None,
        help="Override the environment config directory. Default: bundle env_config.json, "
             f"otherwise {DEFAULT_ENV_CONFIG_DIR}.",
    )
    parser.add_argument("--env_name", type=str, default=None,
                        help="Override the environment config name without .json. "
                             "Default without a bundle config: left_plaif_relative.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--episodes", type=int, default=0, help="0 means loop forever")
    parser.add_argument(
        "--disable_policy_rotation",
        action="store_true",
        help="Debug: zero policy rotation offsets while preserving translation "
        "and gripper. Relative chunks keep the prediction-time orientation; "
        "the action-frame yaw correction still applies to translation.",
    )
    parser.add_argument(
        "--relative_action_frame_yaw_deg",
        type=float,
        default=None,
        help="Rotate relative_tcp position offsets and rotation vectors about "
        "the prediction-time TCP Z axis, in degrees. Default: bundle value or 0.",
    )
    parser.add_argument(
        "--relative_gripper_action_mode",
        choices=("delta", "absolute"),
        default=None,
        help="Gripper values in relative_tcp chunks: 'delta' accumulates from "
        "prediction-time opening; 'absolute' uses each value directly (default). "
        "Units follow gripper_control.input_range (meters for the meter config).",
    )
    parser.add_argument(
        "--relative_progress",
        choices=("timestep", "tcp"),
        default=None,
        help="Legacy fixed-path progress setting (default: train.relative_progress or timestep). "
             "RPC relative execution now converts to deltas and advances by control timestep; "
             "this setting does not change that execution.",
    )
    parser.add_argument(
        "--relative_gripper_progress",
        choices=("tcp", "timestep"),
        default=None,
        help="Legacy fixed-path gripper progress setting. RPC relative-to-delta execution "
        "uses the same control timestep and delay drop for pose and gripper.",
    )
    parser.add_argument(
        "--send_chunk_length", type=int, default=None,
        help="PLAIF poses per publish (1..50); the environment still steps at train.hz. "
             "Defaults to robot.plaif.send_chunk_length, or 1.",
    )
    parser.add_argument(
        "--request_steps",
        type=int,
        default=8,
        help="Request interval in policy action periods for relative_tcp; control steps for delta. "
             "Includes steps dropped for inference delay as well as executed steps.",
    )
    parser.add_argument(
        "--observation_keys",
        nargs="+",
        default=None,
        metavar="KEY",
        help="Only send these top-level keys from the environment observation. "
        "By default, send the observation unchanged.",
    )
    parser.add_argument(
        "--policy_arm",
        choices=["left", "right", "dual"],
        default=None,
        help="Override which arm the policy drives; 'left', 'right' or 'dual'.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Query and inspect actions without creating a robot/camera environment.")
    parser.add_argument("--dry-run-queries", type=int, default=12,
                        help="Maximum server queries in dry-run mode.")
    parser.add_argument("--diagnostics-dir", type=Path, default=None,
                        help="Save dry-run responses and a JSON report here; omitted: no files.")
    parser.add_argument("--rotation-unit", choices=("rad", "deg"), default=None,
                        help="Explicit server rotation-vector unit; no magnitude-based guessing.")
    parser.add_argument("--preprocess-bundle", type=Path, help="Preprocessed output directory or HDF5 episode")
    parser.add_argument("--dataset-path", type=Path,
                        help="HDF5 dataset file/directory to browse in the monitor; defaults to --preprocess-bundle.")
    parser.add_argument("--lighting-check", action="store_true",
                        help="Compare dataset and live camera images before connecting to the policy/robot.")
    parser.add_argument("--lighting-dataset-path", type=Path,
                        help="Lighting reference .pkl/.h5/.hdf5 file or directory; defaults to --dataset-path or --preprocess-bundle.")
    parser.add_argument("--lighting-port", type=int, default=8126,
                        help="Local lighting comparison UI port (0 chooses a free port).")
    parser.add_argument("--lighting-correction", type=Path,
                        help="Reuse a lighting correction JSON (max age: 3 hours). With --lighting-check, "
                             "load it for preview and save the selected correction here on continue.")
    parser.add_argument("--monitor-port", type=int, default=8125, help="Local HTML monitor port (0 chooses a free port)")
    parser.add_argument("--record-dir", type=Path, default=None,
                        help="Save episode records here; omitted: memory-only monitor, no disk recording.")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Disable HTML UI; disk recording requires --record-dir")
    parser.add_argument("--no-browser", action="store_true",
                        help="Keep the HTML monitor running without opening a browser")
    parser.add_argument("--dry-run-hdf5", type=Path, help="Query recorded observations; never construct a robot")
    args = parser.parse_args()
    if args.relative_action_frame_yaw_deg is not None and not np.isfinite(args.relative_action_frame_yaw_deg):
        parser.error("--relative_action_frame_yaw_deg must be finite")
    if args.request_steps < 1:
        parser.error("--request_steps must be at least 1")
    lighting_dataset = args.lighting_dataset_path or args.dataset_path or args.preprocess_bundle
    if args.lighting_dataset_path and not args.lighting_check:
        parser.error("--lighting-dataset-path requires --lighting-check")
    if args.lighting_correction and (args.dry_run or args.dry_run_hdf5):
        parser.error("--lighting-correction cannot be combined with dry-run modes")
    if args.lighting_check:
        if not lighting_dataset:
            parser.error("--lighting-check requires --lighting-dataset-path, --dataset-path or --preprocess-bundle")
        if args.dry_run or args.dry_run_hdf5:
            parser.error("--lighting-check cannot be combined with hardware-free dry-run modes")

    env_cfg_path = None
    if args.preprocess_bundle and args.env_path is None and args.env_name is None:
        bundle_path = args.preprocess_bundle.expanduser()
        bundle_dir = bundle_path.parent if bundle_path.is_file() else bundle_path
        bundled_config = bundle_dir / "env_config.json"
        if bundled_config.is_file():
            env_cfg_path = bundled_config
    if env_cfg_path is None:
        env_cfg_path = (Path(args.env_path or DEFAULT_ENV_CONFIG_DIR).expanduser()
                        / f"{args.env_name or 'left_plaif_relative'}.json")
    print(f"Loading environment config from: {env_cfg_path}")
    cfg = load_config(env_cfg_path)
    if args.relative_progress is None:
        args.relative_progress = cfg.get('train', {}).get('relative_progress', 'timestep')
    if args.relative_progress not in ('tcp', 'timestep'):
        parser.error("train.relative_progress must be tcp or timestep")
    cfg.setdefault('train', {})['relative_progress'] = args.relative_progress
    preprocessor = None
    if args.preprocess_bundle:
        from preprocess.inference import InferencePreprocessor
        preprocessor = InferencePreprocessor(args.preprocess_bundle)
        options = preprocessor.interpreter_options(decoded_gripper=preprocessor.uses_opening_gripper)
        arm = 'dual' if len(options['policy_sides']) == 2 else options['policy_sides'][0]
        if args.policy_arm and args.policy_arm != arm:
            parser.error("--policy_arm differs from the preprocessing bundle")
        if args.observation_keys or args.rotation_unit == 'deg':
            parser.error("Bundle observation keys and radian actions cannot be overridden")
        for name in ('relative_action_frame_yaw_deg', 'relative_gripper_action_mode'):
            if getattr(args, name) is not None and getattr(args, name) != options[name]:
                parser.error(f"--{name} differs from the preprocessing bundle")
        cfg.setdefault("train", {}).update(policy_arm=arm, action_chunk_mode=options['mode'],
                                          policy_action_period_s=options['action_period_s'])
        args.relative_action_frame_yaw_deg = options['relative_action_frame_yaw_deg']
        args.relative_gripper_action_mode = options['relative_gripper_action_mode']
        cfg = preprocessor.execution_config(cfg)
    elif args.policy_arm:
        cfg.setdefault("train", {})["policy_arm"] = args.policy_arm
    else:
        cfg.setdefault('train', {}).setdefault('policy_arm', cfg.get('train', {}).get('arm_type', 'left'))
    if args.relative_action_frame_yaw_deg is None:
        args.relative_action_frame_yaw_deg = 0.0
    if args.relative_gripper_action_mode is None:
        args.relative_gripper_action_mode = 'absolute'
    if args.send_chunk_length is not None:
        cfg.setdefault("robot", {}).setdefault("plaif", {})["send_chunk_length"] = args.send_chunk_length
    send_chunk_length = resolve_send_chunk_length(cfg)

    lighting_correction = None
    correction_path = None
    if args.lighting_check or args.lighting_correction:
        from .inference.lighting_correction import (
            PolicyLightingCorrection, cache_path, load_correction, save_correction,
        )
        correction_path = (args.lighting_correction.expanduser() if args.lighting_correction
                           else cache_path(cfg, preprocessor, lighting_dataset))
        print(f'[Env Client] Lighting session file: {correction_path}', flush=True)
        try:
            lighting_correction = load_correction(correction_path)
            PolicyLightingCorrection(lighting_correction, cfg, preprocessor)
        except (ValueError, OSError) as exc:
            if not args.lighting_check:
                parser.error(str(exc))
            lighting_correction = None
            print(f'[Env Client] Starting a fresh lighting check: {exc}', flush=True)
        if args.lighting_check:
            from .inference.lighting import run_lighting_check
            result = run_lighting_check(cfg, lighting_dataset, port=args.lighting_port,
                no_browser=args.no_browser, initial_correction=lighting_correction, policy_mode=True)
            if not result:
                print('[Env Client] Lighting check cancelled; evaluation not started.', flush=True)
                return
            lighting_correction = result.correction
            if lighting_correction is not None:
                # Reject a reference fitted in a different image domain before robot construction.
                PolicyLightingCorrection(lighting_correction, cfg, preprocessor)
                save_correction(correction_path, lighting_correction)
            else:
                correction_path.unlink(missing_ok=True)
        if lighting_correction:
            print(f'[Env Client] Fixed policy image correction enabled for '
                  f'{sorted(lighting_correction["cameras"])}; fitted at '
                  f'{lighting_correction["created_utc"]}.', flush=True)
        else:
            print('[Env Client] No policy image correction selected.', flush=True)

    client = HTTPPolicyClient(args.trainer_ip, args.trainer_port)
    if preprocessor:
        from .inference.protocol import bundle_id
        client.session.headers['X-Preprocess-Id'] = bundle_id(preprocessor)
    if args.dry_run_hdf5:
        if not preprocessor:
            parser.error("--dry-run-hdf5 requires --preprocess-bundle")
        from .inference.dry_run import run_bundle_dry_run
        try:
            run_bundle_dry_run(args, preprocessor, client)
        finally:
            client.session.close()
        return
    if args.dry_run:
        if preprocessor:
            parser.error("Use --dry-run-hdf5 with a bundle")
        from .inference.diagnostics import run_dry_run
        try:
            run_dry_run(args, cfg, client)
        finally:
            client.session.close()
        return

    server_metadata = client.health()
    print(f"Server health: {server_metadata}")
    if preprocessor:
        from .inference.protocol import validate_server
        validate_server(preprocessor, server_metadata)
        rotation_unit = 'rad'
    else:
        if server_metadata.get('actions') == 'normalized' or server_metadata.get('simulation_only'):
            raise ValueError('This server requires a matching --preprocess-bundle (offline servers require --dry-run-hdf5)')
        _, rotation_unit = resolve_server_contract(cfg, server_metadata, args.rotation_unit)

    active_sides = ('left', 'right') if cfg['train']['arm_type'] == 'dual' else (cfg['train']['arm_type'],)
    if not set(resolve_policy_sides(cfg)).issubset(active_sides):
        raise ValueError(f"Policy sides {resolve_policy_sides(cfg)} unavailable in env {active_sides}")
    if preprocessor:
        from preprocess.convert import StateDecoder
        if (preprocessor.config['actions']['mode'] in ('delta', 'delta_flange') and
                not np.isclose(1 / cfg['train']['hz'], preprocessor.config['action_period_s'])):
            raise ValueError('Delta/delta_flange chunks require env train.hz to match the bundle action period')
        decoder = StateDecoder(cfg)
        for side in resolve_policy_sides(cfg):
            decoder.columns([f'{side}/tcp_pose'])
        for entry in preprocessor.config['observations']:
            if entry['type'] == 'state':
                decoder.columns(entry['keys'])  # Fail before constructing hardware.
    env = build_environment(cfg)

    action_pool = None
    journal = monitor = None
    try:
        policy_sides = resolve_policy_sides(cfg)
        policy_io = PolicyIO(
            config=cfg,
            action_dim=int(env.action_space.shape[0]),
            observation_keys=args.observation_keys,
            policy_sides=policy_sides,
            preprocessor=preprocessor,
            lighting_correction=lighting_correction,
        )
        chunk_interpreter = ChunkInterpreter(
            env,
            mode=resolve_action_chunk_mode(cfg),
            policy_sides=policy_sides,
            relative_progress=args.relative_progress,
            relative_gripper_progress=args.relative_gripper_progress,
            relative_gripper_action_mode=args.relative_gripper_action_mode,
            relative_action_frame_yaw_deg=args.relative_action_frame_yaw_deg,
            disable_policy_rotation=args.disable_policy_rotation,
            action_period_s=cfg["train"].get("policy_action_period_s", 0.1),
        )
        print(
            f"[Env Client] policy arm(s): {policy_sides}, "
            f"action chunk mode: {chunk_interpreter.mode}, "
            f"legacy relative path progress: {chunk_interpreter.relative_progress}, "
            f"policy rotation disabled: {chunk_interpreter.disable_policy_rotation}, "
            f"relative action frame yaw: {chunk_interpreter.relative_action_frame_yaw_deg} deg, "
            f"legacy relative gripper progress: {chunk_interpreter.relative_gripper_progress}, "
            f"relative gripper action mode: {chunk_interpreter.relative_gripper_action_mode}, "
            f"request steps: {args.request_steps}, "
            f"send chunk length: {send_chunk_length}, "
            f"observation keys: {policy_io.observation_keys or 'all'}"
        )
        if chunk_interpreter.mode == 'relative_tcp':
            print('[Env Client] Relative execution: difference full relative plan into global deltas, '
                  'drop delayed control steps, anchor absolute targets once at current measured TCP. '
                  'Legacy relative_progress / relative_gripper_progress do not select TCP tracking.')


        from .inference.monitor import EpisodeJournal, MonitorServer
        from .inference.rollout import run_episode
        if args.record_dir is not None or not args.no_monitor:
            journal = EpisodeJournal(args.record_dir)
        print(f"[Env Client] recording: {args.record_dir if args.record_dir is not None else 'disabled'}", flush=True)
        if not args.no_monitor:
            monitor = MonitorServer(journal, args.monitor_port,
                                    dataset_path=args.dataset_path or args.preprocess_bundle)
            print(f"[Env Client] HTML monitor: http://127.0.0.1:{monitor.port}", flush=True)
            if not args.no_browser:
                monitor.open_browser()
        action_pool = ThreadPoolExecutor(max_workers=1)
        episode = 0
        previous_episode = None
        while True:
            if journal:
                journal.begin(dict(config=cfg, prompt=args.prompt, server=server_metadata,
                                   inference_config=preprocessor.config if preprocessor else None,
                                   normalization=preprocessor.stats if preprocessor else None,
                                   policy_sides=policy_sides, mode=chunk_interpreter.mode,
                                   runtime=dict(disable_policy_rotation=args.disable_policy_rotation,
                                                relative_execution=('delta_fixed_path' if chunk_interpreter.mode == 'relative_tcp' else None),
                                                relative_progress=chunk_interpreter.relative_progress,
                                                relative_gripper_progress=chunk_interpreter.relative_gripper_progress,
                                                relative_gripper_action_mode=args.relative_gripper_action_mode,
                                                lighting_correction=lighting_correction,
                                                lighting_correction_path=str(correction_path) if lighting_correction else None,
                                                relative_action_frame_yaw_deg=args.relative_action_frame_yaw_deg,
                                                request_steps=args.request_steps, send_chunk_length=send_chunk_length),
                                   coordinate_frame='cell-base (m), orientation WXYZ quaternion',
                                   virtual_path='ideal rollout, not a prediction of measured motion'),
                              status='resetting')
            client.reset(args.prompt, previous_episode=previous_episode)
            obs, _ = env.reset()
            previous_episode, quit_requested = run_episode(
                env, obs, client=client, policy_io=policy_io, interpreter=chunk_interpreter,
                pool=action_pool, journal=journal, request_steps=args.request_steps,
                send_chunk_length=send_chunk_length, rotation_unit=rotation_unit)
            episode += 1
            if quit_requested or (args.episodes > 0 and episode >= args.episodes):
                break
        if journal:
            journal.flush()

        print("[Env Client] finished.")
    finally:
        try:
            env.close()
        finally:
            if action_pool is not None:
                action_pool.shutdown(wait=True, cancel_futures=True)
            client.session.close()
            try:
                if monitor:
                    monitor.close()
            finally:
                if journal:
                    if journal.active and journal.status != 'finished':
                        journal.finish('interrupted')
                    journal.close()


if __name__ == "__main__":
    main()
