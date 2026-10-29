"""Exercise GRU conditioning through FLOWER without downloading a VLM."""

from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from flower.models.flower import CausalConv1dHistoryEncoder, FLOWERVLA
from flower.models.networks.force_history import GRUHistoryEncoder


class TinyVLM(nn.Module):
    """Replace only image/text feature extraction for local DiT tests."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(d_model=32))
        self.image_projection = nn.Linear(3, 32)
        self.encoder_projection = nn.Linear(32, 32)

    def _encode_image(self, images):
        return self.image_projection(images.mean(dim=(-1, -2))).unsqueeze(1)

    def get_encoder(self):
        return self.encode

    def encode(self, inputs_embeds, attention_mask):
        return SimpleNamespace(last_hidden_state=self.encoder_projection(inputs_embeds))


def setup_tiny_vlm(model, *args):
    model.vlm = TinyVLM()
    model.prompt_embeds = nn.Parameter(torch.zeros(1, 1, 32), requires_grad=False)
    model.vlm_token_dropout = nn.Identity()


def tiny_text_embeddings(model, text, device):
    return torch.zeros(len(text), 2, 32, device=device)


def make_model(**overrides):
    settings = dict(
        dit_dim=32, n_heads=4, n_layers=2, lowdim_obs_dim=13,
        act_window_size=5, multistep=5, num_sampling_steps=4,
        use_second_view=True, use_proprio=True, use_rope=True,
        attn_pdrop=0.0, resid_pdrop=0.0, mlp_pdrop=0.0,
        use_force_history_adaln=True, force_history_encoder_type="gru",
        force_history_hidden_dim=16, force_history_gru_layers=1,
        force_history_dropout=0.0, force_history_length=50,
    )
    settings.update(overrides)
    with patch.object(FLOWERVLA, "_setup_vlm", setup_tiny_vlm):
        model = FLOWERVLA(**settings)
    model._get_text_embeddings = MethodType(tiny_text_embeddings, model)
    return model.eval()


def make_batch(batch_size=2):
    return {
        "rgb_obs": {key: torch.randn(batch_size, 1, 3, 8, 8)
                    for key in ("rgb_static", "rgb_gripper")},
        "robot_obs": torch.randn(batch_size, 1, 13),
        "right_force_history": torch.randn(batch_size, 1, 50, 6),
        "lang_text": ["Insert USB connector into USB port"] * batch_size,
        "actions": torch.randn(batch_size, 5, 7).clamp(-1, 1),
    }


class ForceHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_gru_preserves_order_and_resets_each_window(self):
        encoder = GRUHistoryEncoder(6, 16, 32).eval()
        history = torch.randn(2, 50, 6)
        expected = encoder(history)
        encoder(torch.randn_like(history))
        torch.testing.assert_close(encoder(history), expected)
        torch.testing.assert_close(
            torch.cat([encoder(window[None]) for window in history]), expected,
        )
        self.assertFalse(torch.allclose(encoder(history.flip(1)), expected))
        # Earlier samples matter even if the newest sample is unchanged.
        self.assertFalse(torch.allclose(encoder(history[:, -1:]), expected))
        self.assertTrue(torch.isfinite(encoder(torch.zeros_like(history))).all())

    def test_gru_rejects_invalid_windows(self):
        encoder = GRUHistoryEncoder(6, 16, 32)
        for shape in ((50, 6), (2, 50, 5), (2, 0, 6), (2, 1, 50, 6)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                encoder(torch.zeros(shape))

    def test_input_shapes_keep_latest_observation_without_normalizing(self):
        model = make_model()
        histories = torch.randn(2, 3, 50, 6)
        prepared = model._normalize_force_history(histories, 2, model.device, torch.float32)
        torch.testing.assert_close(prepared, histories[:, -1], rtol=0, atol=0)
        expected = model.encode_force_history(prepared, 2, model.device, torch.float32)
        torch.testing.assert_close(
            model.encode_force_history(histories, 2, model.device, torch.float32), expected,
        )
        torch.testing.assert_close(
            model.encode_force_history(prepared[0], 1, model.device, torch.float32), expected[:1],
        )

    def test_missing_or_misaligned_histories_fail(self):
        model = make_model()
        with self.assertRaises(KeyError):
            model.encode_observations({k: v for k, v in make_batch().items() if k != "right_force_history"})
        for shape in ((2, 49, 6), (2, 51, 6), (2, 50, 5), (1, 50, 6), (2, 0, 50, 6)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model.encode_force_history(torch.zeros(shape), 2, model.device, torch.float32)

    def test_flow_loss_backpropagates_to_gru_and_adaln(self):
        model = make_model()
        batch = make_batch()
        cond = model.encode_observations(batch)
        self.assertEqual(cond["force_history_embeds"].shape, (2, 32))
        loss, _ = model.rf_loss(cond, batch["actions"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        parameters = [model.force_history_encoder.gru.weight_ih_l0,
                      model.force_history_encoder.gru.weight_hh_l0,
                      model.force_history_encoder.proj.weight,
                      model.adaln["eef_delta"].modCX[-1].weight,
                      model.dit[-1].adaLN_modulation[-1].weight]
        for param in parameters:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0.0)

    def test_changing_only_ft_changes_adaln_and_actions(self):
        model = make_model()
        cond = model.encode_observations(make_batch())
        different = dict(cond, force_history_embeds=model.encode_force_history(
            torch.zeros(2, 50, 6), 2, model.device, torch.float32,
        ))
        signals = []
        hook = model.dit[0].adaLN_modulation.register_forward_hook(
            lambda module, inputs, output: signals.append(output.detach().clone())
        )
        try:
            actions = torch.randn(2, 5, 7)
            time = torch.full((2,), 0.5)
            first = model.dit_forward(actions, time, cond)
            second = model.dit_forward(actions, time, different)
        finally:
            hook.remove()
        self.assertFalse(torch.allclose(signals[0], signals[1]))
        self.assertFalse(torch.allclose(first, second))

    def test_inference_encodes_gru_once_for_all_flow_steps(self):
        model = make_model()
        batch = make_batch(1)
        with patch.object(model.force_history_encoder, "forward", wraps=model.force_history_encoder.forward) as encode:
            with torch.no_grad():
                actions = model(batch, {"lang_text": batch["lang_text"][0]})
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(actions.shape, (1, 5, 7))
        self.assertTrue(torch.isfinite(actions).all())

    def test_direct_dit_raw_history_matches_cached_embedding(self):
        model = make_model()
        cond = model.encode_observations(make_batch())
        raw_cond = {key: value for key, value in cond.items() if key != "force_history_embeds"}
        actions, time = torch.randn(2, 5, 7), torch.full((2,), 0.5)
        torch.testing.assert_close(
            model.dit_forward(actions, time, cond), model.dit_forward(actions, time, raw_cond),
        )

    def test_disabled_ft_runs_without_history(self):
        model = make_model(use_force_history_adaln=False)
        self.assertIsNone(model.force_history_encoder)
        batch = make_batch(1)
        del batch["right_force_history"]
        with torch.no_grad():
            actions = model(batch, {"lang_text": batch["lang_text"][0]})
        self.assertTrue(torch.isfinite(actions).all())

    def test_legacy_conv_and_gru_weights_can_be_restored(self):
        for encoder_type in ("conv1d", "gru"):
            with self.subTest(encoder_type=encoder_type):
                model = make_model(force_history_encoder_type=encoder_type)
                saved_hparams = dict(model.hparams)
                if encoder_type == "conv1d":
                    # Old checkpoints predate the encoder selector and GRU options.
                    for key in ("force_history_encoder_type", "force_history_gru_layers", "force_history_length"):
                        saved_hparams.pop(key)
                with patch.object(FLOWERVLA, "_setup_vlm", setup_tiny_vlm):
                    restored = FLOWERVLA(**saved_hparams).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                history = torch.randn(2, 50, 6)
                torch.testing.assert_close(model.force_history_encoder(history), restored.force_history_encoder(history))
                expected_class = CausalConv1dHistoryEncoder if encoder_type == "conv1d" else GRUHistoryEncoder
                self.assertIsInstance(restored.force_history_encoder, expected_class)


if __name__ == "__main__":
    unittest.main()
