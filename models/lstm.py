# models/lstm.py
# LSTM sequence model.
#
# Long Short-Term Memory with forget + input + output gates.
# Best architecture for long lookback windows (120-day) where the model
# needs to selectively remember information from weeks ago.
#
# Parameter count reference (hidden=128, layers=2):
#   LSTM ≈ 4 × (input_size × hidden + hidden² + hidden) × num_layers

import torch
import torch.nn as nn
from typing import Optional

from models.base import BaseModel


class LSTMModel(BaseModel):
    """
    MIMO LSTM for stock return prediction.

    Input  : (batch, seq_len, input_dim)
    Output : (batch, output_dim)   — all horizon steps at once (no recursion)
    """

    def __init__(self,
                 input_dim:      int,
                 hidden_size:    int,
                 output_dim:     int,
                 num_layers:     int   = 2,
                 dropout:        float = 0.20,
                 use_batch_norm: bool  = False,
                 use_identity:   bool  = False,
                 identity_dim:   int   = 4):
        super().__init__(input_dim, hidden_size, output_dim,
                         use_identity, identity_dim)

        self.lstm = nn.LSTM(
            input_size  = self.effective_input_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size) if use_batch_norm else nn.Identity()
        self.head = _prediction_head(hidden_size, output_dim, dropout)
        _init_lstm_weights(self.lstm)
        _init_head_weights(self.head)

    def forward(self,
                x:        torch.Tensor,
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        x      = self._inject(x, identity)       # (B, T, F+D) or (B, T, F)
        out, _ = self.lstm(x)                    # _ = (h_n, c_n) — unused
        last   = self.norm(out[:, -1, :])
        return self.head(last)


def build_lstm(cfg_model, n_features: int, horizon: int) -> LSTMModel:
    return LSTMModel(
        input_dim      = n_features,
        hidden_size    = cfg_model.hidden_size,
        output_dim     = horizon,
        num_layers     = cfg_model.num_layers,
        dropout        = cfg_model.dropout,
        use_batch_norm = cfg_model.use_batch_norm,
        use_identity   = cfg_model.use_identity,
        identity_dim   = cfg_model.identity_dim,
    )


# ── Shared weight-init helpers (imported by gru.py and rnn.py) ────

def _prediction_head(hidden_size: int, output_dim: int,
                     dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(dropout / 2),
        nn.Linear(hidden_size // 2, output_dim),
    )


def _init_lstm_weights(lstm: nn.LSTM):
    for name, p in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p.data)
        elif "bias" in name:
            p.data.zero_()
            # Forget-gate bias = 1 (helps gradient flow in early training)
            n = p.size(0)
            p.data[n // 4: n // 2].fill_(1.0)


def _init_gru_rnn_weights(module):
    """Xavier / orthogonal init for GRU and RNN (no forget-gate trick needed)."""
    for name, p in module.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p.data)
        elif "bias" in name:
            p.data.zero_()


def _init_head_weights(head: nn.Sequential):
    for layer in head:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
