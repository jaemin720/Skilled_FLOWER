"""Temporal encoders for already-normalized force/torque histories."""

import torch
from torch import nn


class GRUHistoryEncoder(nn.Module):
    """Encode an oldest-to-newest [B,T,D] window, resetting state per call."""

    def __init__(self, input_dim, hidden_dim, out_dim, n_layers=1, dropout=0.0):
        super().__init__()
        if min(input_dim, hidden_dim, out_dim, n_layers) < 1:
            raise ValueError("GRU dimensions and n_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("GRU dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(f"history must be [B,T,{self.input_dim}], got {tuple(history.shape)}")
        if history.shape[1] == 0:
            raise ValueError("force history must contain at least one sample")
        # Omitting h_0 creates zeros for every window; no state crosses batches
        # or demonstrations. Keep all 500 Hz samples in their stored order.
        _, hidden = self.gru(history)
        return self.norm(self.proj(self.dropout(hidden[-1])))
