"""RPC adapter checks use tiny CPU stubs only; never construct FLOWERVLA."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import server_flower_rpc as rpc
from deployment.default_sfm.reference.Real_Robo import env_rpc_server as wire
from deployment.default_sfm.reference.Real_Robo.inference import protocol as client_wire


@pytest.fixture
def bundle():
    entries = [dict(target=k, type=t) for k, t in (
        ("robot_obs", "state"), ("right_force_history", "array"), ("rgb_static", "image"), ("rgb_gripper", "image"))]
    return SimpleNamespace(config=dict(version=1, observations=entries, actions=dict(length=5, sides=["right"])),
                           stats=dict(schema=dict(shapes=dict(robot_obs=[13], right_force_history=[50, 6],
                                                             rgb_static=[224, 224, 3], rgb_gripper=[224, 224, 3]),
                                                  action_shape=[5, 7])))


@pytest.fixture
def observation():
    return dict(robot_obs=np.arange(13, dtype=np.float32),
                right_force_history=np.linspace(-3, 4, 300, dtype=np.float32).reshape(50, 6),
                rgb_static=np.full((224, 224, 3), 51, np.uint8),
                rgb_gripper=np.full((224, 224, 3), 204, np.uint8))


def config(indices):
    transforms = OmegaConf.load(rpc.ROOT / "conf/datamodule/transforms/default_sfm_transforms.yaml")
    return OmegaConf.create(dict(datamodule=dict(transforms=transforms, proprioception_dims=dict(keep_indices=indices))))


@pytest.mark.parametrize("indices", [[[0, 13]], [[0, 1], [3, 13]]])
def test_batch_preserves_standardized_state_and_ft(observation, indices):
    cfg = config(indices)
    selected = rpc.state_indices(cfg)
    batch = rpc.observation_batch(observation, "Insert USB connector into USB port", selected, rpc.image_transforms(cfg))
    np.testing.assert_array_equal(batch["robot_obs"].numpy(), observation["robot_obs"][selected][None, None])
    np.testing.assert_array_equal(batch["right_force_history"].numpy(), observation["right_force_history"][None, None])
    for key, pixel in (("rgb_static", .2), ("rgb_gripper", .8)):
        tensor = batch["rgb_obs"][key]
        assert tensor.shape == (1, 1, 3, 224, 224)
        expected = (torch.tensor(pixel) - torch.tensor([.48145466, .4578275, .40821073])) / torch.tensor([.26862954, .26130258, .27577711])
        torch.testing.assert_close(tensor[0, 0, :, 100, 100], expected)


def test_wire_matches_unchanged_client(bundle, observation):
    client_wire.validate_server(bundle, wire.metadata(bundle))
    for key, value in observation.items():
        np.testing.assert_array_equal(client_wire.unpack_arrays(wire.pack_arrays(observation))[key], value)
        np.testing.assert_array_equal(wire.unpack_arrays(client_wire.pack_arrays(observation))[key], value)
    for horizon in (5, 10, 50):
        actions = np.ones((horizon, 7), np.float32)
        np.testing.assert_array_equal(client_wire.validate_actions(bundle, wire.validate_actions(bundle, actions)), actions)
    wrong = dict(wire.metadata(bundle), bundle_id="wrong")
    with pytest.raises(ValueError, match="bundle_id"):
        client_wire.validate_server(bundle, wrong)


class CPUStub:
    act_window_size = 10
    action_dim = 7
    lowdim_obs_dim = 13
    use_second_view = True
    use_force_history_adaln = True
    force_history_length = 50

    def reset(self):
        self.was_reset = True

    def encode_observations(self, batch):
        self.batch = batch
        return batch

    def sample_actions(self, noise, features, inference=False):
        assert inference and noise.device.type == "cpu"
        return noise.clamp(-1, 1)


def test_policy_returns_full_checkpoint_horizon(bundle, observation):
    model = CPUStub()
    policy = rpc.FlowerRPCPolicy(model, config([[0, 13]]), bundle,
                                 {"usb": "Insert USB connector into USB port"}, device="cpu")
    with pytest.raises(ValueError, match="/reset"):
        policy.predict(observation)
    with pytest.raises(ValueError, match="trained task"):
        policy.reset("unknown instruction")
    policy.reset("usb")
    actions = policy.predict(observation)
    assert model.was_reset and actions.shape == (10, 7) and actions.dtype == np.float32
    assert model.batch["lang_text"] == ["Insert USB connector into USB port"]
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match="right_force_history"):
        policy.predict(dict(observation, right_force_history=np.zeros((49, 6))))


def test_strict_ema_loading():
    original = dict(a=torch.zeros(2), b=torch.zeros(1))
    ema = dict(a=torch.ones(2), b=torch.ones(1))
    ckpt = dict(state_dict=original, global_step=7, callbacks={"EMA": dict(cur_step=7, ema_weights=list(ema.values()))})
    assert rpc.checkpoint_state(ckpt, original, "ema")["a"] is ema["a"]
    assert rpc.checkpoint_state(ckpt, original, "raw")["a"] is original["a"]
    with pytest.raises(ValueError, match="order"):
        rpc.checkpoint_state(ckpt, dict(reversed(list(original.items()))), "ema")
    ckpt["callbacks"]["EMA"]["cur_step"] = 6
    with pytest.raises(ValueError, match="EMA"):
        rpc.checkpoint_state(ckpt, original, "ema")


def test_prepared_bundle():
    path = rpc.ROOT / "datasets/default_sfm_crop_v2/rpc_bundle"
    if not path.exists():
        pytest.skip("Production bundle is generated next to the local dataset")
    bundle, manifest = rpc.verify_bundle(path)
    assert manifest["normalization_refitted"] is False
    assert bundle.config["actions"]["gripper"]["label_version"] == 1
    assert bundle.stats["actions"]["groups"]["gripper"]["method"] == "none"
