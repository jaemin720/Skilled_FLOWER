"""Regression for EMA with Quest's int32 quantizer buffers."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytorch_lightning as pl
import pytest
import torch

from flower.callbacks.ema import EMA


class BufferedModel(pl.LightningModule):
    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.register_buffer("discrete", torch.tensor([0, 1], dtype=dtype))
        self.register_buffer("running_mean", torch.tensor([2.0]))


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool])
def test_ema_preserves_discrete_buffers_and_checkpoint_roundtrip(dtype):
    model = BufferedModel(dtype)
    ema = EMA(decay=0.999, save_ema_weights_in_callback_state=True)
    ema.log = Mock()
    ema.on_train_start(None, model)
    with torch.no_grad():
        model.weight.fill_(10.0)
        model.running_mean.fill_(6.0)
        model.discrete.copy_(torch.tensor([1, 0], dtype=dtype))
    ema.on_train_batch_end(SimpleNamespace(global_step=3), model, None, None, 2)
    decay = ema.get_decay(3)
    expected_weight = torch.tensor([2.0 * decay + 10.0 * (1 - decay)])
    expected_mean = torch.tensor([2.0 * decay + 6.0 * (1 - decay)])
    # Live model stays untouched; EMA keeps the complete state-dict ordering.
    assert model.weight.item() == 10.0
    restored = EMA(decay=0.999, save_ema_weights_in_callback_state=True)
    restored.load_state_dict(ema.state_dict())
    restored.on_train_start(None, model)
    restored.replace_model_weights(model)
    torch.testing.assert_close(model.weight, expected_weight)
    torch.testing.assert_close(model.running_mean, expected_mean)
    assert model.discrete.dtype == dtype
    assert torch.equal(model.discrete, torch.tensor([1, 0], dtype=dtype))
    restored.restore_original_weights(model)
    assert model.weight.item() == 10.0
    assert model.running_mean.item() == 6.0
