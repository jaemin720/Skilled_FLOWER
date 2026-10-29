#!/usr/bin/env python3
"""CPU-only check using Kingston's existing preprocessor. Never opens a robot or server."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, bundle, fixture, provenance):
    sys.path.insert(0, str(root))
    from preprocess.inference import InferencePreprocessor
    from Real_Robo.inference.protocol import (
        bundle_id, metadata, pack_arrays, unpack_arrays,
        validate_actions, validate_observation, validate_server,
    )

    source = json.loads(provenance.read_text())
    for name, expected in source["files"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Kingston source changed since preparation: {name}")
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(bundle / name) != expected:
            raise ValueError(f"Bundle copy differs: {name}")
    preprocessor = InferencePreprocessor(bundle)
    assert bundle_id(preprocessor) == manifest["bundle_id"]
    validate_server(preprocessor, metadata(preprocessor))
    collection = json.loads((root / "Real_Robo/configs/default_sfm.json").read_text())
    assert preprocessor.uses_opening_gripper is False
    assert preprocessor.execution_config(collection) is collection
    assert preprocessor.interpreter_options() == dict(
        mode="delta", policy_sides=("right",), relative_gripper_action_mode="absolute",
        relative_action_frame_yaw_deg=0., action_period_s=.1)
    with np.load(fixture, allow_pickle=False) as data:
        cases = len(data["raw__state"])
        errors = {key.removeprefix("expected__"): 0. for key in data.files if key.startswith("expected__")}
        for i in range(cases):
            raw = {key.removeprefix("raw__"): data[key][i] for key in data.files if key.startswith("raw__")}
            # Actual client supplies its live collection config to the decoder.
            actual = validate_observation(preprocessor, preprocessor.observation(raw, collection_config=collection))
            for key, value in actual.items():
                expected = data["expected__" + key][i]
                np.testing.assert_array_equal(value, expected, err_msg=f"case={i}, {key}")
                errors[key] = max(errors[key], float(np.max(np.abs(value.astype(float) - expected))))
            restored = unpack_arrays(pack_arrays(actual))
            for key in actual:
                np.testing.assert_array_equal(actual[key], restored[key])
        actions = validate_actions(preprocessor, data["normalized_actions"])
        physical = preprocessor.compose_actions(preprocessor.denormalize_actions(actions))
        np.testing.assert_allclose(physical, data["physical_actions"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(preprocessor.normalize_actions(physical), actions, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(physical[:, 6], actions[:, 6])
        np.testing.assert_array_equal(unpack_arrays(pack_arrays({"actions": actions}))["actions"], actions)
    return dict(passed=True, bundle_id=bundle_id(preprocessor), observations_tested=cases,
                observation_max_abs_error=errors, action_horizon_tested=len(actions), gripper_labels_unchanged=True,
                source_files_unchanged=len(source["files"]),
                client_sha256=sha256(root / "Real_Robo/env_rpc_client.py"),
                model_loaded=False, gpu_used=False, robot_started=False, http_server_started=False,
                scope="Synthetic preprocessing and wire format only; no model or hardware execution")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-share-root", type=Path, default=Path("/home/csilab/share_projects/clean_share"))
    parser.add_argument("--bundle", type=Path, default=here / "rpc_bundle")
    parser.add_argument("--fixture", type=Path, default=here / "verification.npz")
    parser.add_argument("--provenance", type=Path, default=here / "source_provenance.json")
    parser.add_argument("--output", type=Path, default=here / "client_bundle_check.json")
    args = parser.parse_args()
    report = verify(args.clean_share_root, args.bundle, args.fixture, args.provenance)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
