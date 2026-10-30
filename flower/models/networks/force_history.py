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


class FAFMPGRUHistoryEncoder(nn.Module):
    """Port of FAFMP/utils/networks.py:TemporalGRUEncoder for [B,T,D].

    FAFMP applies the reset gate BEFORE the candidate hidden projection;
    torch.nn.GRU applies it after that projection and is not equivalent.
    The sequence starts at zero for each observation window. No normalization
    or dropout is added to the already-standardized F/T values or final latent.
    """

    def __init__(self, input_dim=6, hidden_dim=128, out_dim=128):
        super().__init__()
        if min(input_dim, hidden_dim, out_dim) < 1:
            raise ValueError("FAFMP GRU dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.update_x = nn.Linear(hidden_dim, hidden_dim)
        self.update_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_x = nn.Linear(hidden_dim, hidden_dim)
        self.reset_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cand_x = nn.Linear(hidden_dim, hidden_dim)
        self.cand_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(2 * hidden_dim, out_dim)

    @property
    def recurrent_weight(self):
        """Weight tracked by TrainingStatus to verify optimizer updates."""
        return self.update_x.weight

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(f"history must be [B,T,{self.input_dim}], got {tuple(history.shape)}")
        if history.shape[1] == 0:
            raise ValueError("force history must contain at least one sample")
        x = self.activation(self.input_proj(history))
        # Batch the input projections, keeping the reference recurrent equations.
        update_inputs, reset_inputs, candidate_inputs = self.update_x(x), self.reset_x(x), self.cand_x(x)
        hidden = x.new_zeros(x.shape[0], self.hidden_dim)
        hidden_history = []
        for step in range(x.shape[1]):
            update = torch.sigmoid(update_inputs[:, step] + self.update_h(hidden))
            reset = torch.sigmoid(reset_inputs[:, step] + self.reset_h(hidden))
            candidate = torch.tanh(candidate_inputs[:, step] + self.cand_h(reset * hidden))
            hidden = (1.0 - update) * candidate + update * hidden
            hidden_history.append(hidden)
        mean_hidden = torch.stack(hidden_history, dim=1).mean(dim=1)
        return self.proj(torch.cat([hidden, mean_hidden], dim=-1))


class FAFMPConditioningMLP(nn.Module):
    """Concat + activated MLP, adapted to FLOWER's AdaLN output width.

    Matches FAFMP's MLP(activate_final=True, layer_norm=False) structure and
    Flax GELU's default tanh approximation. Widths are chosen for FLOWER.
    """

    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        if min(input_dim, hidden_dim, out_dim) < 1:
            raise ValueError("Conditioning MLP dimensions must be positive")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(approximate="tanh"),
        )

    def forward(self, *conditions: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat(conditions, dim=-1))
