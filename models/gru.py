# models/gru.py
# GRU sequence model.
#
# Gated Recurrent Unit — one update gate + one reset gate.
# ~33% fewer parameters than LSTM at the same hidden size.
# Trains ~25% faster. Often matches LSTM quality on shorter windows (30-day).
#
# Parameter count reference (hidden=128, layers=2):
#   GRU ≈ 3 × (input_size × hidden + hidden² + hidden) × num_layers

import torch
import torch.nn as nn
from typing import Optional

from models.base import BaseModel
from models.lstm import _prediction_head, _init_gru_rnn_weights, _init_head_weights


class GRUModel(BaseModel):
    """
    MIMO GRU for stock return prediction.

    Input  : (batch, seq_len, input_dim)
    Output : (batch, output_dim)
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

        self.gru = nn.GRU(
            input_size  = self.effective_input_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size) if use_batch_norm else nn.Identity()
        self.head = _prediction_head(hidden_size, output_dim, dropout)
        _init_gru_rnn_weights(self.gru)
        _init_head_weights(self.head)

    def forward(self,
                x:        torch.Tensor,
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        x      = self._inject(x, identity)
        out, _ = self.gru(x)                         # _ = h_n — unused
        last   = self.norm(out[:, -1, :])
        return self.head(last)


def build_gru(cfg_model, n_features: int, horizon: int) -> GRUModel:
    return GRUModel(
        input_dim      = n_features,
        hidden_size    = cfg_model.hidden_size,
        output_dim     = horizon,
        num_layers     = cfg_model.num_layers,
        dropout        = cfg_model.dropout,
        use_batch_norm = cfg_model.use_batch_norm,
        use_identity   = cfg_model.use_identity,
        identity_dim   = cfg_model.identity_dim,
    )
