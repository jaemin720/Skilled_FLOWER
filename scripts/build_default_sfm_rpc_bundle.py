#!/usr/bin/env python3
"""Package existing SFM statistics for the unchanged clean_share RPC client."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from deployment.default_sfm.reference.Real_Robo.env_rpc_server import InferenceBundle, bundle_id


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bundle(dataset, environment, output):
    dataset, environment, output = map(Path, (dataset, environment, output))
    if output.exists():
        raise FileExistsError(f"Bundle already exists: {output}; select a new version")
    source_stats = json.loads((dataset / "normalization.json").read_text())
    contract = json.loads((dataset / "data_contract.json").read_text())
    plan = json.loads((dataset / "preprocessing_config.json").read_text())
    manifest = json.loads((dataset / "manifest.json").read_text())
    if not manifest["complete"] or manifest["smoke_test"]:
        raise ValueError("Use the complete production dataset")
    if sha256(dataset / "normalization.json") != contract["normalization_sha256"]:
        raise ValueError("Source normalization hash differs from the dataset contract")
    collection = json.loads(environment.read_text())
    expected = json.loads((dataset / "raw_inventory.json").read_text())["collection"]
    for key in ("arm_type", "hz", "proprio_keys"):
        if collection["train"][key] != expected["train"][key]:
            raise ValueError(f"Environment differs from training: {key}")
    if collection["env"]["right"]["gripper_control"]["mode"] != "binary":
        raise ValueError("SFM held gripper labels require the original binary control mode")
    if collection.get("gripper_interfaces") is None:
        collection.pop("gripper_interfaces", None)
    if collection.get("image_preprocess", {}).get("enable", False):
        raise ValueError("Native images are required before bundle preprocessing")
    collection["train"].update(policy_arm="right", action_chunk_mode="delta", policy_action_period_s=0.1)
    pose_components = ("x", "y", "z", "roll", "pitch", "yaw")
    state_keys = ["right/gripper_pose", dict(key="right/tcp_pose",
                  components={key: True for key in pose_components}, wrap_angles=["roll", "yaw"]), "right/tcp_vel"]
    observations = [dict(source="state", target="robot_obs", type="state", keys=state_keys)]
    for camera, target in (("head_cam_d455", "rgb_static"), ("wrist_cam_d405_a", "rgb_gripper")):
        observations.append(dict(source=camera, target=target, type="image", input_color="rgb",
                                 crop=plan["vision"]["crop_tlhw"][camera], resize=[224, 224],
                                 resize_mode="letterbox", interpolation="area"))
    observations.append(dict(source="right_force_history", target="right_force_history", type="array", sample_ndim=2))
    # This length records the source preprocessing recipe. The wire protocol
    # explicitly allows a trained policy to return a different horizon.
    action = dict(mode="delta", interpreter_mode="delta", length=plan["action"]["chunk_length"],
                  sides=["right"], source_format="absolute", action_period_s=0.1,
                  rotation_unit="rad", translation_unit="m", rotation_representation="rotvec",
                  pose_reference="observations/state at each timestep", relative_action_frame_yaw_deg=0.0,
                  relative_gripper_action_mode="absolute", padding="zero_after_normalization", padding_mask="action_mask",
                  gripper=dict(label_version=1, mode="absolute", source_mode="binary", command_range=[-1., 1.],
                               observation_range=[0., 1.], offset=0.02, deadband=0.001, grasp=dict(enabled=False)))
    config = dict(version=1, observations=observations, collection_config=collection,
                  actions=action, normalization_file="normalization.json", action_period_s=0.1)
    columns = ["right/gripper_pose/pos", *["right/tcp_pose/" + k for k in pose_components],
               *["right/tcp_vel/" + k for k in ("vx", "vy", "vz", "wx", "wy", "wz")]]
    schema = dict(state_columns={"robot_obs": columns}, observations=observations,
                  shapes=dict(robot_obs=[13], rgb_static=[224, 224, 3], rgb_gripper=[224, 224, 3], right_force_history=[50, 6]),
                  action_shape=[action["length"], 7], action=action)
    groups = {}
    for name, indices in (("xyz", [0, 1, 2]), ("rpy", [3, 4, 5])):
        coefficients = deepcopy(source_stats["action_pose"])
        for key in ("mean", "std", "min", "max", "offset", "scale"):
            coefficients[key] = [coefficients[key][i] for i in indices]
        groups[name] = dict(coefficients, indices=indices)
    groups["gripper"] = dict(method="none", indices=[6], offset=[0.], scale=[1.], epsilon=1e-6, temperature=20.)
    stats = dict(version=1, schema=schema,
                 observations=dict(robot_obs=deepcopy(source_stats["state"]), right_force_history=deepcopy(source_stats["ft"])),
                 actions=dict(method="grouped", epsilon=1e-6, temperature=20., groups=groups,
                              offset=source_stats["action_pose"]["offset"] + [0.],
                              scale=source_stats["action_pose"]["scale"] + [1.]))
    output.mkdir(parents=True)
    write_json(output / "inference_config.json", config)
    write_json(output / "normalization.json", stats)
    write_json(output / "env_config.json", collection)
    write_json(output / "instructions.json", contract["instruction_mapping"])
    for source, target in (("normalization.json", "source_normalization.json"),
                           ("preprocessing_config.json", "source_preprocessing_config.json"),
                           ("data_contract.json", "source_data_contract.json")):
        shutil.copyfile(dataset / source, output / target)
    loaded = InferenceBundle(output)
    record = dict(version=1, bundle_id=bundle_id(loaded), source_dataset=str(dataset.resolve()),
                  source_normalization_sha256=sha256(dataset / "normalization.json"),
                  files={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
                  normalization_refitted=False, normalized_on_client=True,
                  gripper_labels="held command target: closed=-1, open=+1; legacy label_version=1",
                  dataset_action_horizon=action["length"], inference_action_horizon="from checkpoint",
                  server_model_loaded=False, robot_started=False)
    write_json(output / "bundle_manifest.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "datasets/default_sfm_crop_v2")
    parser.add_argument("--environment", type=Path,
                        default=REPO / "deployment/default_sfm/reference/Real_Robo/configs/default_sfm.json")
    parser.add_argument("--output", type=Path, default=REPO / "datasets/default_sfm_crop_v2/rpc_bundle")
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.dataset, args.environment, args.output), indent=2))


if __name__ == "__main__":
    main()
