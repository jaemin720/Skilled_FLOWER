"""Responsive rollout loop: UI commands are consumed only at control boundaries."""
import copy
import time
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import TimeoutError
import numpy as np
from .monitor import virtual_paths, observation_forces
from rb_env.envs.action_utils import hold_action


def measured_poses(env, sides):
    return {side: np.asarray(getattr(env.unwrapped, f'env_{side}').currpos, dtype=float).copy()
            for side in sides}


def stop_tail(env):
    stop = getattr(env.unwrapped, 'stop_action_chunk', None)
    if stop:
        stop()


@contextmanager
def episode_clock(env, paused):
    """Keep observation/manual-control cycles out of episode counters while paused."""
    if not paused:
        yield
        return
    base = env.unwrapped
    targets = [base] + [arm for side in ('left', 'right')
                        if (arm := getattr(base, f'env_{side}', None)) is not None]
    missing = object()
    previous = [(target, vars(target).get('episode_clock_paused', missing)) for target in targets]
    try:
        for target, _ in previous:
            target.episode_clock_paused = True
        yield
    finally:
        for target, value in previous:
            if value is missing:
                del target.episode_clock_paused
            else:
                target.episode_clock_paused = value


@dataclass
class Prediction:
    future: object = None
    request_id: str = None
    discarded: bool = False

    def discard(self, journal):
        self.discarded = True
        if self.future is not None:
            self.future.cancel()
        if journal and self.request_id:
            journal.discarded(self.request_id)


def run_episode(env, initial_observation, *, client, policy_io, interpreter, pool,
                journal=None, request_steps=8, send_chunk_length=1, rotation_unit='rad',
                start_paused=False):
    from .policy_io import response_to_chunk, compensate_chunk_delay, episode_result
    sides = interpreter.policy_sides
    obs = initial_observation
    t, ptr, delay = 0, 0, 0
    request_offset = 0
    current, pending = None, None
    retired = []
    paused = bool(start_paused or (journal and journal.pause_requested))
    intervened = False
    phase = None
    reward, terminated, truncated, info = 0., False, False, {}
    outcome, quit_requested = None, False
    last_recorded = -1
    hz = getattr(env.unwrapped, f'env_{sides[0]}').hz
    interpreter.begin_episode()

    def submit(observation, timestep):
        entry = Prediction()
        # Observation and anchor refer to the same control boundary, before any
        # next env.step. All transforms, RPC, plotting geometry run off-thread.
        raw = copy.deepcopy(observation)
        poses = (policy_io.observation_poses(raw) if policy_io.preprocessor
                 else measured_poses(env, sides))
        context = interpreter.capture_context()
        if context is not None and policy_io.preprocessor:
            context.update(poses)
        gripper_context = None
        if policy_io.preprocessor and policy_io.preprocessor.uses_opening_gripper:
            previous = [getattr(getattr(env.unwrapped, f'env_{side}'), '_last_gripper_target', None) for side in sides]
            previous = [float(value) / 100. for value in previous] if all(value is not None for value in previous) else None
            gripper_context = policy_io.gripper_context(raw, previous)
        episode = journal.active if journal else None

        def predict():
            transmitted = policy_io.observation(raw)
            if journal:
                entry.request_id = journal.prediction(episode, timestep, transmitted, poses)
            try:
                response = client.infer(transmitted)
                normalized = response_to_chunk(response, 'rad' if policy_io.preprocessor else rotation_unit)
                physical = policy_io.physical_actions(normalized)
                execution = policy_io.execution_actions(physical, gripper_context)
                effective = execution.copy()
                if interpreter.disable_policy_rotation:
                    for i in range(len(sides)):
                        effective[:, i*7+3:i*7+6] = 0
                chunk = interpreter.prepare_execution(policy_io.to_env_chunk(execution), context)
                if journal:
                    paths = virtual_paths(effective, poses, sides, interpreter.mode,
                                          interpreter.relative_action_frame_yaw_deg)
                    journal.response(entry.request_id, normalized, physical, paths,
                                     execution_actions=chunk.actions)
                return chunk
            except Exception as exc:
                if journal and entry.request_id:
                    journal.response(entry.request_id, None, None, {}, repr(exc))
                raise
            finally:
                if entry.discarded and journal and entry.request_id:
                    journal.discarded(entry.request_id)
        entry.future = pool.submit(predict)
        return entry

    def discard_pending():
        nonlocal pending
        if pending:
            pending.discard(journal)
            retired.append(pending)
            pending = None

    def record(command=None, step_info=None, state='waiting'):
        nonlocal last_recorded
        if journal:
            poses = (policy_io.observation_poses(obs) if policy_io.preprocessor
                     else measured_poses(env, sides))
            journal.frame(t, obs, poses, command=command, info=step_info, phase=state,
                          replace=t == last_recorded, elapsed_s=t / hz,
                          forces=observation_forces(obs, policy_io.config, sides))
            last_recorded = t

    try:
        while True:
            if journal:
                journal.check()
                for command in journal.pop_controls():
                    if command in ('success', 'failure', 'skip'):
                        outcome = command
                        break
                    if command == 'pause' and not paused:
                        paused = True
                        stop_tail(env)
                        discard_pending()
                        current = None
                    elif command == 'resume' and paused:
                        paused = False
                        # Refresh context after pausing; never resume a stale plan.
                        discard_pending()
                        current = None
                        interpreter.begin_episode()
                if outcome:
                    break
            # Old in-flight work may finish, but can never be activated again.
            retired[:] = [p for p in retired if not p.future.done()]
            if pending and pending.future.done() and not paused and not intervened:
                predicted = pending.future.result()
                current, dropped = compensate_chunk_delay(
                    predicted, delay, drop_exhausted=interpreter.mode == 'relative_tcp')
                if current is not None:
                    current = interpreter.anchor_execution(current)
                if current is not None and current.paths:
                    interpreter.activate(current, 0 if current.fixed_control_path else delay)
                if journal:
                    if current is None:
                        journal.discarded(pending.request_id, dropped)
                    else:
                        journal.applied(pending.request_id, t, dropped,
                                        execution_paths=current.paths if current.fixed_control_path else None)
                pending = None
                ptr, delay = 0, 0
                # Dropped control cycles already consume the new chunk's
                # request interval; its retained execution still starts at 0.
                request_offset = dropped

            complete = (current is not None and
                        (interpreter.is_complete(current) if current.paths else
                         ptr >= len(current)))
            if complete:
                current = None
            threshold = request_steps
            # Arms are the authority for control frequency in DualRb3Env.
            if current is not None and interpreter.mode == 'relative_tcp':
                hz = getattr(interpreter.env.unwrapped, f'env_{sides[0]}').hz
                # Match prepare_execution's control-time conversion: floating
                # error (3 * .1 * 10 > 3) must not delay prefetch by one cycle.
                threshold = max(1, int(np.ceil(request_steps * interpreter.action_period_s * hz - 1e-9)))
            if (not paused and not intervened and pending is None and not retired and
                    (current is None or request_offset + ptr >= threshold)):
                pending = submit(obs, t)
                delay = 0
                # Let a ready local response replace the plan before another
                # command. The bounded wait cannot hide UI controls for an RPC timeout.
                try:
                    pending.future.result(timeout=.002)
                except TimeoutError:
                    pass
                else:
                    continue
            new_phase = 'paused' if paused else ('running' if current is not None and not intervened else 'waiting')
            if new_phase != phase:
                phase = new_phase
                if journal:
                    journal.set_status(phase)
            width = send_chunk_length
            action = (interpreter.command_for_chunk(current, ptr, width)
                      if new_phase == 'running' else hold_action(env))
            # Paused cycles refresh the current step instead of advancing its clock.
            record(action, info, phase)
            with episode_clock(env, paused):
                next_obs, reward, terminated, truncated, info = env.step(action)
            if journal:
                journal.command_result(t, info, sides)
            obs = next_obs
            if not paused:
                t += 1
            quit_requested = bool(info.get('keyboard', {}).get('quit_requested'))
            if quit_requested or terminated or truncated:
                break
            manual = 'intervene_action' in info or 'action_intervene' in info
            if manual:
                discard_pending()
                current = None
            elif intervened:
                current = None
                interpreter.begin_episode()
            intervened = manual
            if new_phase == 'running':
                ptr += 1
                if pending:
                    delay += 1
    finally:
        stop_tail(env)
        discard_pending()
        record(step_info=info, state='final')
        # Drain in-flight requests before reset/save: they remain visible as
        # discarded chunks attached to their original observation timestep.
        for entry in retired:
            if not entry.future.cancelled():
                try:
                    entry.future.result()
                except Exception:
                    pass
    result = episode_result(reward, terminated, truncated, info)
    if outcome:
        result.update(outcome=outcome, terminated=outcome in ('success','failure'), truncated=outcome == 'skip')
    if journal:
        journal.finish(result['outcome'])
    return result, quit_requested
