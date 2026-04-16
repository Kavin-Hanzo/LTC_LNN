# models/rnn.py
# Vanilla RNN sequence model.
#
# No gating — just a single tanh recurrent transformation.
# Minimal parameter count; prone to vanishing gradients beyond ~20 steps.
# Use as a weak architectural baseline to show that gating (GRU/LSTM)
# and continuous-time dynamics (LNN) each contribute value.
#
# Parameter count reference (hidden=128, layers=2):
#   RNN ≈ (input_size × hidden + hidden² + hidden) × num_layers
#       ≈ 1/4 of LSTM at the same hidden size

import torch
import torch.nn as nn
from typing import Optional

from models.base import BaseModel
from models.lstm import _prediction_head, _init_gru_rnn_weights, _init_head_weights


class RNNModel(BaseModel):
    """
    MIMO vanilla RNN for stock return prediction.

    Input  : (batch, seq_len, input_dim)
    Output : (batch, output_dim)

    Note: with lookback=120 this model will likely suffer from
    vanishing gradients. Recommended only for lookback ≤ 30.
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

        self.rnn = nn.RNN(
            input_size   = self.effective_input_dim,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            nonlinearity = "tanh",
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size) if use_batch_norm else nn.Identity()
        self.head = _prediction_head(hidden_size, output_dim, dropout)
        _init_gru_rnn_weights(self.rnn)
        _init_head_weights(self.head)

    def forward(self,
                x:        torch.Tensor,
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        x      = self._inject(x, identity)
        out, _ = self.rnn(x)                         # _ = h_n — unused
        last   = self.norm(out[:, -1, :])
        return self.head(last)


def build_rnn(cfg_model, n_features: int, horizon: int) -> RNNModel:
    return RNNModel(
        input_dim      = n_features,
        hidden_size    = cfg_model.hidden_size,
        output_dim     = horizon,
        num_layers     = cfg_model.num_layers,
        dropout        = cfg_model.dropout,
        use_batch_norm = cfg_model.use_batch_norm,
        use_identity   = cfg_model.use_identity,
        identity_dim   = cfg_model.identity_dim,
    )
