"""Policy plans with timestep or measured-progress execution; emits absolute XYZ/RPY."""
from __future__ import annotations
from copy import copy
import numpy as np
from scipy.spatial.transform import Rotation
from rb_env.envs.action_commands import PreparedChunk
from rb_env.envs.action_utils import resolve_side_slices, hold_action, pose_to_rpy, rpy_to_pose
from rb_env.envs.plaif_math import (rotation_from_wxyz, rotation_to_wxyz, bounded_target,
                                   absolute_targets_to_flange_deltas)
from rb_env.utils.scipy_spatial_utils import apply_local_pose_offset_rotvec, apply_global_pose_offset_rotvec

DELTA = "delta"
DELTA_FLANGE = "delta_flange"
RELATIVE_TCP = "relative_tcp"


def interpolate_pose(a, b, fraction):
    f = float(np.clip(fraction, 0., 1.))
    ra, rb = rotation_from_wxyz(a[3:]), rotation_from_wxyz(b[3:])
    r = ra * Rotation.from_rotvec((ra.inv() * rb).as_rotvec() * f)
    return np.concatenate((a[:3] + f * (b[:3] - a[:3]), rotation_to_wxyz(r)))


def sample_path(path, u):
    u = float(np.clip(u, 0., len(path)-1))
    i = min(int(u), len(path)-2)
    return interpolate_pose(path[i], path[i+1], u-i)


def motion_distance(a, b):
    return (float(np.linalg.norm(b[:3]-a[:3])),
            float((rotation_from_wxyz(a[3:]).inv()*rotation_from_wxyz(b[3:])).magnitude()))


def path_budget(path, start, stop):
    """Arc length and angular travel across ALL segments crossed this cycle."""
    knots = [start] + [float(i) for i in range(int(np.floor(start))+1, int(np.ceil(stop)))
                        if start < i < stop] + [stop]
    total = np.zeros(2)
    for a, b in zip(knots, knots[1:]):
        total += motion_distance(sample_path(path, a), sample_path(path, b))
    return total




class ChunkInterpreter:
    def __init__(self, env, mode, policy_sides, relative_gripper_progress=None,
                 relative_gripper_action_mode="delta", relative_action_frame_yaw_deg=0.,
                 disable_policy_rotation=False, action_period_s=0.1,
                 position_tolerance_m=0.001, rotation_tolerance_rad=0.01,
                 settle_speed_mps=0.01, settle_speed_radps=0.1,
                 relative_progress="timestep"):
        if mode not in (DELTA, DELTA_FLANGE, RELATIVE_TCP):
            raise ValueError(f"Unsupported action chunk mode: {mode!r}")
        if relative_progress not in ("tcp", "timestep"):
            raise ValueError("relative_progress must be 'tcp' or 'timestep'")
        if relative_gripper_progress is None:
            relative_gripper_progress = relative_progress
        if relative_gripper_progress not in ("tcp", "timestep"):
            raise ValueError("relative_gripper_progress must be 'tcp' or 'timestep'")
        if relative_gripper_action_mode not in ("delta", "absolute"):
            raise ValueError("relative_gripper_action_mode must be 'delta' or 'absolute'")
        for name, value in dict(action_period_s=action_period_s,
                                position_tolerance_m=position_tolerance_m,
                                rotation_tolerance_rad=rotation_tolerance_rad,
                                settle_speed_mps=settle_speed_mps,
                                settle_speed_radps=settle_speed_radps).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        if not np.isfinite(relative_action_frame_yaw_deg):
            raise ValueError("relative_action_frame_yaw_deg must be finite")
        if mode == DELTA_FLANGE and relative_action_frame_yaw_deg != 0:
            raise ValueError('delta_flange uses native flange axes; relative_action_frame_yaw_deg must be 0')
        self.env, self.mode = env, mode
        self.policy_sides = tuple(policy_sides)
        self.relative_progress = relative_progress
        self.relative_gripper_progress = relative_gripper_progress
        self.relative_gripper_action_mode = relative_gripper_action_mode
        self.relative_action_frame_yaw_deg = float(relative_action_frame_yaw_deg)
        self.disable_policy_rotation = bool(disable_policy_rotation)
        self._relative_action_to_tcp = Rotation.from_euler("z", relative_action_frame_yaw_deg, degrees=True)
        self.action_dim = int(env.action_space.shape[0])
        self._side_envs, self._slices = {}, {}
        for side in self.policy_sides:
            arm = getattr(env.unwrapped, f"env_{side}", None)
            if arm is None:
                raise ValueError(f"side {side!r} is not available in the environment")
            if not np.isfinite(arm.hz) or arm.hz <= 0:
                raise ValueError("arm hz must be finite and positive")
            self._side_envs[side] = arm
            self._slices[side] = resolve_side_slices(self.action_dim, side)
        self.begin_episode()

    def begin_episode(self):
        self._active_chunk = None
        self._progress = {}
        self._issued = {}
        self._cycles = 0
        self._elapsed_control_steps = 0

    def capture_context(self):
        if self.mode in (DELTA, DELTA_FLANGE):
            return None
        context = {}
        for side, arm in self._side_envs.items():
            context[side] = np.asarray(arm.currpos, dtype=float).copy()
            if self.relative_gripper_action_mode == "delta":
                lo, hi = arm.gripper_continuous_input_range
                out_lo, out_hi = arm.gripper_continuous_output_range
                opening = float(arm.curr_gripper_pos)*100.
                context[f"{side}_gripper"] = np.array([np.clip(
                    lo+(opening-out_lo)/(out_hi-out_lo)*(hi-lo), lo, hi)])
        return context

    def _checked_actions(self, actions):
        actions = np.asarray(actions, dtype=float).copy()
        if actions.ndim != 2 or not len(actions) or actions.shape[1] != self.action_dim or not np.all(np.isfinite(actions)):
            raise ValueError(f"actions must be finite with shape [T, {self.action_dim}]")
        if self.disable_policy_rotation:
            for _, rot, _ in self._slices.values():
                actions[:, rot] = 0.
        return actions

    def prepare_execution(self, actions, context=None):
        """Decode RPC relative targets into control-period world deltas.

        Relative positions share the request TCP axes; rotation vectors encode
        orientations relative to that same fixed frame. Difference the complete
        plan BEFORE dropping latency, without ever using the request position.
        After latency dropping, anchor_execution integrates the retained deltas
        once at the activation pose and fixes all future absolute targets.
        """
        if self.mode in (DELTA, DELTA_FLANGE):
            return self.prepare(actions, context)
        actions = self._checked_actions(actions)
        if context is None:
            raise ValueError("relative_tcp chunks require prediction-time axes")
        horizon = len(actions)
        cycles = {side: max(1, int(np.ceil(horizon * self.action_period_s * arm.hz - 1e-9)))
                  for side, arm in self._side_envs.items()}
        deltas = np.repeat(actions[:1], max(cycles.values()), axis=0)
        for side, arm in self._side_envs.items():
            pos, rot, grip = self._slices[side]
            request_axes = rotation_from_wxyz(np.asarray(context[side])[3:])
            positions = self._relative_action_to_tcp.apply(actions[:, pos])
            rotations = Rotation.from_rotvec(self._relative_action_to_tcp.apply(actions[:, rot]))
            local = np.vstack((np.array([0., 0., 0., 1., 0., 0., 0.]),
                               np.column_stack((positions, rotations.as_quat(scalar_first=True)))))
            gripper = float(context.get(f"{side}_gripper", [0.])[0])
            grips = []
            for value in actions[:, grip]:
                value += gripper if self.relative_gripper_action_mode == 'delta' else 0.
                gripper = float(np.clip(value, *arm.gripper_continuous_input_range))
                grips.append(gripper)
            advance = 1. / arm.hz / self.action_period_s
            before = local[0]
            for k in range(len(deltas)):
                after = sample_path(local, (k + 1) * advance)
                deltas[k, pos] = request_axes.apply(after[:3] - before[:3])
                # Global rotation deltas left-compose. Subtracting rotvec
                # components is incorrect for rotations about different axes.
                dr = rotation_from_wxyz(after[3:]) * rotation_from_wxyz(before[3:]).inv()
                deltas[k, rot] = request_axes.apply(dr.as_rotvec())
                deltas[k, grip] = grips[min(int(k * advance + 1e-9), horizon - 1)]
                before = after
        return PreparedChunk(deltas, action_period_s=self.action_period_s, policy_horizon=horizon)

    def anchor_execution(self, chunk):
        """Fix a latency-trimmed relative plan at the current pose exactly once.

        Rows already represent control cycles. Neither latency nor policy-period
        interpolation is applied again while executing these absolute targets.
        Limits belong to the backend: clipping one send must not move the next
        planned target, so tracking error carries into the next command.
        """
        if self.mode != RELATIVE_TCP or chunk.paths:
            return chunk
        paths = {}
        for side, arm in self._side_envs.items():
            pos, rot, _ = self._slices[side]
            pose = np.asarray(arm.currpos, dtype=float).copy()
            path = [pose]
            for action in chunk.actions:
                pose = apply_global_pose_offset_rotvec(pose, action[pos.start:rot.stop].copy())
                path.append(pose)
            paths[side] = np.stack(path)
        return PreparedChunk(chunk.actions, paths, chunk.action_period_s,
                             chunk.policy_horizon, fixed_control_path=True)

    def prepare(self, actions, context=None):
        """Decode the original plan for fixed-anchor path inspection/tracking."""
        actions = self._checked_actions(actions)
        if self.mode in (DELTA, DELTA_FLANGE):
            return PreparedChunk(actions, action_period_s=self.action_period_s)
        if context is None:
            raise ValueError("relative_tcp chunks require prediction-time poses")
        paths = {}
        for side, arm in self._side_envs.items():
            pos, rot, grip = self._slices[side]
            base = np.asarray(context[side], dtype=float)
            path = [base]
            gripper = float(context.get(f"{side}_gripper", [0.])[0])
            for action in actions:
                offset = np.concatenate((self._relative_action_to_tcp.apply(action[pos]),
                                         self._relative_action_to_tcp.apply(action[rot])))
                path.append(apply_local_pose_offset_rotvec(base, offset))
                value = action[grip] + (gripper if self.relative_gripper_action_mode == "delta" else 0.)
                gripper = float(np.clip(value, *arm.gripper_continuous_input_range))
                action[grip] = gripper
            paths[side] = np.stack(path)
        return PreparedChunk(actions, paths, self.action_period_s)

    def activate(self, chunk, elapsed_control_steps=0):
        """Account for inference delay by control time or a geometry search window."""
        if elapsed_control_steps < 0:
            raise ValueError("elapsed_control_steps must be non-negative")
        self._activate(chunk)
        self._elapsed_control_steps = elapsed_control_steps
        for side, arm in self._side_envs.items():
            self._issued[side] = min(len(chunk), (elapsed_control_steps+1)/arm.hz/chunk.action_period_s)

    def _activate(self, chunk):
        if self._active_chunk is chunk:
            return
        self._active_chunk = chunk
        self._progress = {s: 0. for s in self.policy_sides}
        # Legacy tracked paths search only their first control window, never a
        # distant crossing. Fixed control paths index their target rows directly.
        self._issued = {s: min(len(chunk), 1./a.hz/chunk.action_period_s)
                        for s, a in self._side_envs.items()}
        self._cycles = 0
        self._elapsed_control_steps = 0

    def _project(self, path, current, old, upper, advance):
        best_u, best_error = old, float("inf")
        for i in range(min(int(old), len(path)-2), min(int(upper)+1, len(path)-1)):
            a, b = path[i:i+2]
            if i > int(old):
                # A nearby following segment cannot bypass an unfinished
                # rotation at the preceding corner.
                previous = rotation_from_wxyz(path[i-1, 3:])
                turn = (previous.inv()*rotation_from_wxyz(a[3:])).as_rotvec()
                magnitude = np.linalg.norm(turn)
                if magnitude > 1e-12:
                    done = (previous.inv()*rotation_from_wxyz(current[3:])).as_rotvec()
                    if np.dot(done, turn)/magnitude < magnitude-self.rotation_tolerance_rad:
                        continue
            dp = b[:3]-a[:3]
            ra = rotation_from_wxyz(a[3:])
            dr = (ra.inv()*rotation_from_wxyz(b[3:])).as_rotvec()
            travelled = (ra.inv()*rotation_from_wxyz(current[3:])).as_rotvec()
            parts = []
            if np.dot(dp, dp) > 1e-16:
                parts.append(np.dot(current[:3]-a[:3], dp)/np.dot(dp, dp))
            if np.dot(dr, dr) > 1e-16:
                parts.append(np.dot(travelled, dr)/np.dot(dr, dr))
            if parts:
                alpha = float(np.clip(min(parts), 0., 1.))
                u = float(np.clip(i+alpha, old, upper))
            else:
                dist, angle = motion_distance(current, a)
                # Dwell/gripper-only segments advance by CONTROL time only
                # after their TCP pose is reached, not by projection ties.
                u = min(i+1., old+advance, upper) if (dist <= self.position_tolerance_m and angle <= self.rotation_tolerance_rad) else max(old, float(i))
            end_dist, end_angle = motion_distance(current, b)
            if (i+1 <= upper and end_dist <= self.position_tolerance_m
                    and end_angle <= self.rotation_tolerance_rad and parts):
                u = max(u, float(i+1))
            projected = sample_path(path, u)
            dist, angle = motion_distance(current, projected)
            error = dist**2 + (0.05*angle)**2
            if error < best_error - 1e-15:
                best_u, best_error = u, error
        return max(old, best_u)

    def command_for_step(self, chunk, step_index):
        poses = {s: np.asarray(a.currpos, dtype=float).copy() for s, a in self._side_envs.items()}
        return self._command_for_poses(chunk, step_index, poses)

    def command_for_chunk(self, chunk, step_index, length):
        """Advance one real cycle, then preview on isolated progress/pose state.

        A length of one preserves the existing one-dimensional action contract.
        Predicted poses only choose future targets; PlaifNode still computes all
        wire deltas against fresh feedback at publish time.
        """
        if type(length) is not int or not 1 <= length <= 50:
            raise ValueError("send chunk length must be an integer in [1, 50]")
        if length == 1:
            return self.command_for_step(chunk, step_index)
        if not chunk.paths or chunk.fixed_control_path:
            length = min(length, len(chunk) - step_index)
            if length < 1:
                raise ValueError("step index is past the policy chunk")
        elif self.relative_progress == "timestep":
            self._activate(chunk)
            total_steps = max(int(np.ceil(len(chunk) * chunk.action_period_s * arm.hz - 1e-9))
                              for arm in self._side_envs.values())
            length = min(length, total_steps - self._elapsed_control_steps - step_index)
            if length < 1:
                raise ValueError("step index is past the policy chunk")
        poses = {s: np.asarray(a.currpos, dtype=float).copy() for s, a in self._side_envs.items()}
        actions = [self._command_for_poses(chunk, step_index, poses)]
        preview = copy(self)
        preview._progress = self._progress.copy()
        preview._issued = self._issued.copy()
        for offset in range(1, length):
            for side, arm in self._side_envs.items():
                pos, rot, _ = self._slices[side]
                target = rpy_to_pose(actions[-1][pos.start:rot.stop])
                # Estimate the bounded command pose, never mutate live feedback.
                delta = absolute_targets_to_flange_deltas(
                    poses[side], [target],
                    max_translation_step=getattr(arm, "max_translation_speed_mps", 0.05) / arm.hz,
                    max_rotation_step=getattr(arm, "max_rotation_speed_radps", 0.1) / arm.hz,
                )[0]
                rotation = rotation_from_wxyz(poses[side][3:])
                poses[side] = np.concatenate((
                    poses[side][:3] + rotation.apply(delta[:3]),
                    rotation_to_wxyz(rotation * Rotation.from_rotvec(delta[3:])),
                ))
            actions.append(preview._command_for_poses(chunk, step_index + offset, poses))
        # Only widen the search window to the path actually offered for playback.
        # Observed progress and gripper/control-cycle counters remain unchanged.
        for side in self._issued:
            self._issued[side] = max(self._issued[side], preview._issued[side])
        return np.stack(actions)

    def _command_for_poses(self, chunk, step_index, poses):
        action = hold_action(self.env)
        if chunk.fixed_control_path:
            if not 0 <= step_index < len(chunk):
                raise ValueError("step index is past the policy chunk")
            self._activate(chunk)
            for side in self._side_envs:
                pos, rot, grip = self._slices[side]
                action[pos.start:rot.stop] = pose_to_rpy(chunk.paths[side][step_index + 1])
                action[grip] = chunk.actions[step_index, grip]
            self._cycles = max(self._cycles, step_index + 1)
            return action
        if not chunk.paths:
            raw = chunk.actions[min(step_index, len(chunk)-1)].copy()
            for side, arm in self._side_envs.items():
                pos, rot, grip = self._slices[side]
                apply_offset = apply_local_pose_offset_rotvec if self.mode == DELTA_FLANGE else apply_global_pose_offset_rotvec
                target = apply_offset(poses[side], raw[pos.start:rot.stop])
                action[pos.start:rot.stop] = pose_to_rpy(target)
                action[grip] = raw[grip]
            return action
        self._activate(chunk)
        if self.relative_progress == "timestep":
            return self._command_for_timestep(chunk, step_index, poses, action)
        for side, arm in self._side_envs.items():
            path = chunk.paths[side]
            current = poses[side]
            dt = 1./arm.hz
            advance = dt/chunk.action_period_s
            u = self._project(path, current, self._progress[side], self._issued[side], advance if self._cycles else 0.)
            self._progress[side] = u
            look = min(float(len(chunk)), u+advance)
            self._issued[side] = max(self._issued[side], look)
            target = sample_path(path, look)
            budget = path_budget(path, u, look)
            # A stationary channel still needs a bounded way to recover
            # cross-track error and settle the final pose.
            if budget[0] < 1e-12:
                budget[0] = self.settle_speed_mps*dt
            if budget[1] < 1e-12:
                budget[1] = self.settle_speed_radps*dt
            target = bounded_target(current, target, *budget)
            pos, rot, grip = self._slices[side]
            action[pos.start:rot.stop] = pose_to_rpy(target)
            g = u if self.relative_gripper_progress == "tcp" else self._cycles*advance
            gi = min(int(g+1e-9), len(chunk)-1)
            # Preserve discrete gripper events; a width threshold is not an
            # orientation/position interpolation coordinate.
            action[grip] = chunk.actions[gi, grip]
        self._cycles += 1
        return action

    def _command_for_timestep(self, chunk, step_index, poses, action):
        """Recover timed targets without adding a tracking/settling motion limit.

        At matching control/policy periods, step k emits path[k+1] and gripper[k].
        Different control periods sample the same plan in control time. The
        environment/backend applies the same limits as during data collection.
        """
        for side, arm in self._side_envs.items():
            advance = 1. / arm.hz / chunk.action_period_s
            u = (self._elapsed_control_steps + step_index) * advance
            look = min(float(len(chunk)), u + advance)
            path = chunk.paths[side]
            pos, rot, grip = self._slices[side]
            action[pos.start:rot.stop] = pose_to_rpy(sample_path(path, look))
            g = u
            if self.relative_gripper_progress == "tcp":
                g = self._project(path, poses[side], self._progress[side],
                                  self._issued[side], advance if self._cycles else 0.)
                self._progress[side] = g
            self._issued[side] = max(self._issued[side], look)
            gi = min(int(g + 1e-9), len(chunk) - 1)
            action[grip] = chunk.actions[gi, grip]
        self._cycles = max(self._cycles, step_index + 1)
        return action

    def is_complete(self, chunk):
        if not chunk.paths or self._active_chunk is not chunk:
            return False
        if chunk.fixed_control_path:
            return self._cycles >= len(chunk)
        if self.relative_progress == "timestep":
            elapsed = self._elapsed_control_steps + self._cycles
            return all(elapsed / arm.hz / chunk.action_period_s >= len(chunk) - 1e-9
                       for arm in self._side_envs.values())
        for side, arm in self._side_envs.items():
            if self._issued[side] < len(chunk)-1e-9:
                return False
            dist, angle = motion_distance(arm.currpos, chunk.paths[side][-1])
            if dist > self.position_tolerance_m or angle > self.rotation_tolerance_rad:
                return False
            # Do not finish a loop merely because its endpoint equals its start.
            if self._progress[side] < len(chunk)-1e-9:
                return False
        return True
