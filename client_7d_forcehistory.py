import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import pickle
import sys

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.config import load_config, build_environment


POLICY_DIM = 7
ENV_DIM = 14

RIGHT_GRIPPER_IDX = 19
RIGHT_TCP_FORCE_SLICE = slice(20, 23)
RIGHT_TCP_POSE_SLICE = slice(23, 29)
RIGHT_TCP_POSE_Z_RPY_SLICE = slice(25, 29)
RIGHT_TCP_TORQUE_SLICE = slice(29, 32)
RIGHT_TCP_POSE_ONLY = slice(23, 29)


def as_policy_chunk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        if x.shape[0] != POLICY_DIM:
            raise ValueError(f"Unexpected 1D action shape {x.shape}")
        return x[None, :]
    if x.ndim == 2 and x.shape[-1] != POLICY_DIM and x.shape[0] == POLICY_DIM:
        x = x.T
    if x.shape[-1] != POLICY_DIM:
        raise ValueError(f"Unexpected action shape {x.shape}")
    return x


def pad_action_to_env(x: np.ndarray) -> np.ndarray:
    x = as_policy_chunk(x)
    if x.shape[-1] == ENV_DIM:
        return x
    pad = np.zeros((*x.shape[:-1], ENV_DIM - POLICY_DIM), dtype=np.float32)
    return np.concatenate([pad, x], axis=-1)


def extract_right_state(raw_state: np.ndarray, state_mode: str) -> np.ndarray:
    raw_state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if raw_state.shape[0] != 38:
        raise ValueError(f"Expected raw state dim 38, got {raw_state.shape[0]}")

    if state_mode in {"ft13", "full"}:
        return np.concatenate(
            [
                raw_state[RIGHT_TCP_POSE_SLICE],
                raw_state[[RIGHT_GRIPPER_IDX]],
                raw_state[RIGHT_TCP_FORCE_SLICE],
                raw_state[RIGHT_TCP_TORQUE_SLICE],
            ],
            axis=0,
        ).astype(np.float32)

    if state_mode == "ft11":
        return np.concatenate(
            [
                raw_state[RIGHT_TCP_POSE_Z_RPY_SLICE],
                raw_state[[RIGHT_GRIPPER_IDX]],
                raw_state[RIGHT_TCP_FORCE_SLICE],
                raw_state[RIGHT_TCP_TORQUE_SLICE],
            ],
            axis=0,
        ).astype(np.float32)

    if state_mode == "pose7":
        return np.concatenate(
            [
                raw_state[RIGHT_TCP_POSE_ONLY],
                raw_state[[RIGHT_GRIPPER_IDX]],
            ],
            axis=0,
        ).astype(np.float32)

    raise ValueError(f"Unsupported state_mode: {state_mode}")


def get_obs_key(obs: dict, *keys: str):
    for key in keys:
        if key in obs:
            return obs[key]
    raise KeyError(f"Missing observation key. Tried: {keys}")


def extract_right_force_history(obs: dict) -> np.ndarray:
    history = get_obs_key(
        obs,
        "right_force_history",
        "right_ft_history",
        "force_history",
    )
    history = np.asarray(history, dtype=np.float32)
    while history.ndim > 2 and history.shape[0] == 1:
        history = history[0]

    if history.ndim == 1 and history.size % 6 == 0:
        history = history.reshape(-1, 6)

    if history.ndim != 2 or history.shape[-1] != 6:
        raise ValueError(
            "Expected right_force_history shape [T,6] or [1,T,6], "
            f"got {history.shape}"
        )
    return np.ascontiguousarray(history, dtype=np.float32)


def npify_obs(obs: dict, state_mode: str, use_wrist: bool) -> dict:
    if not isinstance(obs, dict):
        raise TypeError("Observation must be a dict.")

    head = np.asarray(
        get_obs_key(obs, "left/head_cam", "head_cam", "front_cam"),
        dtype=np.uint8,
    ).squeeze(0)
    state = extract_right_state(np.asarray(obs["state"], dtype=np.float32).squeeze(0), state_mode)
    right_force_history = extract_right_force_history(obs)

    out = {
        "rgb_static": head,
        "robot_obs": state,
        "image_primary": head,
        "proprio": state,
        "right_force_history": right_force_history,
    }
    if use_wrist:
        wrist = np.asarray(
            get_obs_key(obs, "right/wrist_cam", "right_wrist_cam"),
            dtype=np.uint8,
        ).squeeze(0)
        out["rgb_gripper"] = wrist
        out["image_wrist"] = wrist
    return out


class FlowerHTTPPolicyClient:
    def __init__(self, host: str, port: int, transport: str = "pickle"):
        self.base_url = f"http://{host}:{port}"
        self.transport = transport
        self.session = requests.Session()

    def reset(self, text: str):
        resp = self.session.post(f"{self.base_url}/reset", json={"text": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health(self):
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def infer(self, observation: dict):
        if self.transport == "pickle":
            return self.infer_pickle(observation)
        return self.infer_json(observation)

    def infer_json(self, observation: dict):
        def convert(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, dict):
                return {k: convert(v) for k, v in x.items()}
            if isinstance(x, list):
                return [convert(v) for v in x]
            return x

        observation = convert(observation)

        resp = self.session.post(
            f"{self.base_url}/query",
            json={"observation": observation},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def infer_pickle(self, observation: dict):
        payload = pickle.dumps(observation, protocol=pickle.HIGHEST_PROTOCOL)
        resp = self.session.post(
            f"{self.base_url}/query_pickle",
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=60,
        )
        resp.raise_for_status()
        actions = np.load(BytesIO(resp.content), allow_pickle=False)
        return {"actions": actions}
XYZ_SCALE = 1.0
RPY_SCALE = 1.0
GRIPPER_SCALE = 1.0
def response_to_chunk(response: dict) -> np.ndarray:
    if not isinstance(response, dict):
        raise TypeError(f"Response must be dict, got {type(response)!r}")

    if "actions" not in response:
        raise KeyError(f"Missing 'actions' in response. Response keys: {list(response.keys())}")

    actions = response["actions"]

    if isinstance(actions, dict):
        raise TypeError(
            "response['actions'] is still a dict. "
            "Use --transport pickle, or decode JSON with json_numpy.loads(resp.text). "
            f"actions keys={list(actions.keys())}"
        )

    actions = as_policy_chunk(np.asarray(actions, dtype=np.float32))

    

    # raw_action 모델용: tiny clipping 제거
    # 필요하면 안전하게만 [-1, 1] 범위로 제한
    actions[:, :6] = np.clip(actions[:, :6], -1.0, 1.0)
    actions[:, 6] = np.clip(actions[:, 6], -1.0, 1.0)

    

    return actions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_ip", type=str, default="115.145.175.14")
    parser.add_argument("--trainer_port", type=int, default=45587)
    parser.add_argument("--env_path", type=str, default="/home/csilab/jhlee_workspace/mo_hil-serl/Real_Robo/experiments/configs/", help="e.g., 'usb'")
    parser.add_argument("--env_name", type=str, default="default_lsm", help="e.g., 'usb'")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=0, help="0 means loop forever")
    parser.add_argument("--action_delay", type=int, default=1)
    parser.add_argument("--action_chunk", type=int, default=4)
    parser.add_argument(
        "--state_mode",
        choices=["pose7", "ft11", "ft13", "full"],
        default="pose7",
        help=(
            "Use 'pose7' for [x,y,z,r,p,y,gripper], which the force-history "
            "server slices to [z,r,p,y,gripper]."
        ),
    )
    wrist_group = parser.add_mutually_exclusive_group()
    wrist_group.add_argument(
        "--use_wrist",
        action="store_true",
        dest="use_wrist",
        help="Send wrist image to server.",
    )
    wrist_group.add_argument(
        "--no_use_wrist",
        action="store_false",
        dest="use_wrist",
        help="Do not send wrist image to server.",
    )
    parser.set_defaults(use_wrist=True)
    parser.add_argument(
        "--transport",
        choices=["pickle", "json"],
        default="pickle",
        help="Use the binary pickle/npy path for lower overhead. 'json' keeps the old list-based path.",
    )
    parser.add_argument(
        "--debug_force_history_prints",
        type=int,
        default=3,
        help="Print force-history payload details for the first N inference requests. Use 0 to disable.",
    )
    args = parser.parse_args()

    env_cfg_path = f"{args.env_path}/{args.env_name}.json"
    print(f"Loading environment config from: {env_cfg_path}")
    cfg = load_config(env_cfg_path)
    env = build_environment(cfg)

    client = FlowerHTTPPolicyClient(args.trainer_ip, args.trainer_port, transport=args.transport)
    try:
        print(f"Server health: {client.health()}")
    except Exception as exc:
        print(f"Warning: health check failed: {exc}")
    client.reset(args.prompt)
    action_pool = ThreadPoolExecutor(max_workers=1)
    debug_counter = {"force_history": 0}

    def predict_chunk(raw_obs):
        obs_payload = npify_obs(
            raw_obs,
            state_mode=args.state_mode,
            use_wrist=args.use_wrist,
        )
        if debug_counter["force_history"] < args.debug_force_history_prints:
            fh = obs_payload["right_force_history"]
            print(
                "[CLIENT force_history] "
                f"shape={fh.shape}, dtype={fh.dtype}, "
                f"last={np.array2string(fh[-1], precision=4, suppress_small=True)}, "
                f"min={fh.min():.4f}, max={fh.max():.4f}",
                flush=True,
            )
            debug_counter["force_history"] += 1

        response = client.infer(obs_payload)
        policy_chunk = response_to_chunk(response)
        env_chunk = pad_action_to_env(policy_chunk)

        # if debug_counter["n"] < 5:
        #     print("========== DEBUG INFER ==========")
        #     print("[state_mode]", args.state_mode)
        #     print("[proprio shape]", obs_payload["proprio"].shape)
        #     print("[proprio]", obs_payload["proprio"])
        #     print("[right_force_history shape]", obs_payload["right_force_history"].shape)
        #     print("[policy_chunk shape]", policy_chunk.shape)
        #     print("[policy first]", policy_chunk[0])
        #     print("[policy min/max]", policy_chunk.min(), policy_chunk.max())
        #     print("[env first 0:7]", env_chunk[0, :7])
        #     print("[env first 7:14]", env_chunk[0, 7:])
        #     print("=================================")
        #     debug_counter["n"] += 1

        return env_chunk
    # def predict_chunk(raw_obs):
    #     return pad_action_to_env(
    #         response_to_chunk(client.infer(npify_obs(raw_obs, args.state_mode, args.use_wrist)))
    #     )

    def async_predict(raw_obs):
        return action_pool.submit(
            lambda: predict_chunk(raw_obs)
        )
    episode = 0

    while True:
        obs, _ = env.reset()
        current_chunk = predict_chunk(obs)
        ptr = 0
        future = None
        prev_intervened = False
        done = False

        while not done:
            if current_chunk.ndim != 2 or current_chunk.shape[-1] != ENV_DIM:
                raise ValueError(f"Expected action chunk with shape [T, {ENV_DIM}], got {current_chunk.shape}")

            action = current_chunk[ptr]
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            intervened = "action_intervene" in step_info
            done = bool(terminated or truncated)

            if intervened:
                current_chunk = np.zeros_like(current_chunk)
                prev_intervened = True
                continue

            if prev_intervened:
                current_chunk = predict_chunk(next_obs)
                future = None
                ptr = 0
                prev_intervened = False
                obs = next_obs
                continue

            if len(current_chunk) == 1:
                current_chunk = predict_chunk(next_obs)
                ptr = 0
                obs = next_obs
                continue

            action_horizon = max(1, len(current_chunk) - args.action_chunk + 1)

            if future is None and ptr >= len(current_chunk) - action_horizon - args.action_delay:
                future = async_predict(next_obs)

            if ptr == len(current_chunk) - action_horizon:
                if future is None:
                    future = async_predict(next_obs)
                new_chunk = future.result()
                future = None
                current_chunk = new_chunk[args.action_delay:]
                if len(current_chunk) == 0:
                    current_chunk = new_chunk[-1:]
                ptr = 0
                obs = next_obs
                continue

            ptr += 1
            obs = next_obs

        episode += 1
        if args.episodes > 0 and episode >= args.episodes:
            break

    env.close()
    action_pool.shutdown(wait=True)
    print("[Env Client] finished.")


if __name__ == "__main__":
    main()
