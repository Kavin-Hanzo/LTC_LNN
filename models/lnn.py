# models/lnn.py
# Liquid Neural Network (LNN) using a Liquid Time-Constant (LTC) cell.
#
# Source: adapted from the uploaded lnn.py.
# Changes from original:
#   1. LiquidNN now inherits BaseModel instead of plain BaseModel
#   2. forward() accepts (x, identity=None) — identity injected via _inject()
#   3. build_lnn() reads from our ModelConfig dataclass (not a raw dict)
#   4. model_summary() extended with BaseModel fields
#   5. num_layers removed — LNN uses a single LTC cell; depth is expressed
#      via ode_unfolds (finer integration) not stacked layers
#
# WHY LNN IS DIFFERENT FROM LSTM/GRU/RNN:
#
#   Standard RNNs are DISCRETE:   h_t = f(h_{t-1}, x_t)
#   LNN is CONTINUOUS-TIME:       dh/dt = (-h/τ) + tanh(W_in·x + W_rec·h)
#
#   The Euler approximation of this ODE per time-step is:
#       v   = tanh( W_in · x_t  +  W_rec · h )
#       dh  = (-h / τ)  +  v
#       h   = h + dt · dh
#
#   Running this `ode_unfolds` times per observed time-step gives a
#   finer numerical integration without needing more observed data.
#
# PARAMETER TRADE-OFFS vs other archs:
#   LNN has NO separate gate matrices — just W_in and W_rec.
#   Fewer parameters than LSTM/GRU, but richer temporal dynamics.
#   Especially effective when price dynamics have oscillatory / decaying structure.
#
# LNN-specific config (ModelConfig.lnn):
#   tau_constant  : membrane time-constant τ
#                   high τ → state decays slowly (long memory)
#                   low  τ → state decays fast  (reactive to recent input)
#   dt            : Euler step size — smaller = more accurate ODE approximation
#                   typical range 0.01–0.5
#   ode_unfolds   : how many Euler steps per observed time-step
#                   more unfolds = finer integration but slower forward pass
#                   typical range 4–12

import torch
import torch.nn as nn
from typing import Optional

from models.base import BaseModel
from models.lstm import _prediction_head, _init_head_weights


# ── LTC Cell ──────────────────────────────────────────────────────

class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) cell.
    Implements one Euler step of the continuous-time ODE.

    State update:
        v   = tanh( W_in · x  +  W_rec · h )
        dh  = ( -h / tau )  +  v
        h'  = h  +  dt · dh
    """

    def __init__(self, input_size: int, hidden_size: int,
                 tau: float = 1.0, dt: float = 0.1):
        super().__init__()
        self.tau = tau
        self.dt  = dt
        self.input_map    = nn.Linear(input_size,  hidden_size)
        self.recurrent_map = nn.Linear(hidden_size, hidden_size)
        self.tanh         = nn.Tanh()

        # Xavier init for both linear layers
        nn.init.xavier_uniform_(self.input_map.weight)
        nn.init.zeros_(self.input_map.bias)
        nn.init.xavier_uniform_(self.recurrent_map.weight)
        nn.init.zeros_(self.recurrent_map.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, input_size)
        h : (batch, hidden_size)
        → h_new : (batch, hidden_size)
        """
        v  = self.tanh(self.input_map(x) + self.recurrent_map(h))
        dh = (-h / self.tau) + v
        return h + self.dt * dh


# ── Liquid Neural Network ─────────────────────────────────────────

class LiquidNN(BaseModel):
    """
    Liquid Neural Network wrapping the LTCCell.

    Sequence processing:
      for each time-step t in [0, seq_len):
          for each ode_unfold in [0, ode_unfolds):
              h = LTCCell(x_t, h)
      predict from final h

    Input  : (batch, seq_len, input_dim)
    Output : (batch, output_dim)
    """

    def __init__(self,
                 input_dim:    int,
                 hidden_size:  int,
                 output_dim:   int,
                 tau:          float = 1.0,
                 dt:           float = 0.1,
                 ode_unfolds:  int   = 6,
                 dropout:      float = 0.2,
                 use_identity: bool  = False,
                 identity_dim: int   = 4):
        super().__init__(input_dim, hidden_size, output_dim,
                         use_identity, identity_dim)

        # LNN-specific attributes
        self.tau         = tau
        self.dt          = dt
        self.ode_unfolds = ode_unfolds

        self.cell    = LTCCell(self.effective_input_dim, hidden_size, tau, dt)
        self.dropout = nn.Dropout(dropout)
        self.head    = _prediction_head(hidden_size, output_dim, dropout)
        _init_head_weights(self.head)

    def forward(self,
                x:        torch.Tensor,
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x        : (B, T, input_dim)
        identity : (B, identity_dim) or None
        returns  : (B, output_dim)
        """
        x = self._inject(x, identity)               # (B, T, effective_input_dim)

        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            for _ in range(self.ode_unfolds):
                h = self.cell(x[:, t, :], h)        # one Euler step

        h = self.dropout(h)
        return self.head(h)                          # (B, output_dim)

    def model_summary(self) -> dict:
        base = super().model_summary()
        base.update({
            "tau":         self.tau,
            "dt":          self.dt,
            "ode_unfolds": self.ode_unfolds,
        })
        return base


# ── Factory ───────────────────────────────────────────────────────

def build_lnn(cfg_model, n_features: int, horizon: int) -> LiquidNN:
    """
    Build LiquidNN from a ModelConfig dataclass.
    LNN-specific parameters come from cfg_model.lnn (LNNConfig).
    """
    return LiquidNN(
        input_dim    = n_features,
        hidden_size  = cfg_model.hidden_size,
        output_dim   = horizon,
        tau          = cfg_model.lnn.tau_constant,
        dt           = cfg_model.lnn.dt,
        ode_unfolds  = cfg_model.lnn.ode_unfolds,
        dropout      = cfg_model.dropout,
        use_identity = cfg_model.use_identity,
        identity_dim = cfg_model.identity_dim,
    )
