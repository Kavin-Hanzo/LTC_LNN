import torch
import torch.nn as nn
from models.base import BaseModel


# ── LTC Cell ──────────────────────────────────────────────────────────────────

class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Cell.
    Approximates a continuous-time ODE with Euler steps.

    State update:
        v   = tanh( W_in * x + W_rec * h )
        dh  = (-h / tau) + v
        h'  = h + dt * dh
    """

    def __init__(self, input_size: int, hidden_size: int, tau: float = 1.0, dt: float = 0.1):
        super(LTCCell, self).__init__()
        self.tau = tau
        self.dt = dt
        self.input_map = nn.Linear(input_size, hidden_size)
        self.recurrent_map = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        """
        Args:
            x: (batch, input_size)
            h: (batch, hidden_size)
        Returns:
            h_new: (batch, hidden_size)
        """
        v = self.tanh(self.input_map(x) + self.recurrent_map(h))
        dh = (-h / self.tau) + v
        return h + self.dt * dh


# ── Liquid Neural Network ──────────────────────────────────────────────────────

class LiquidNN(BaseModel):
    """
    Liquid Neural Network wrapping the LTCCell.

    Processes a full sequence by:
      1. Iterating over each timestep t
      2. Unfolding the ODE `ode_unfolds` times per timestep (finer Euler integration)
      3. Using the final hidden state to predict forecast_horizon steps

    Input:  (batch, seq_len, input_dim)
    Output: (batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        tau: float = 1.0,
        dt: float = 0.1,
        ode_unfolds: int = 6,
        dropout: float = 0.2,
    ):
        super(LiquidNN, self).__init__(input_dim, hidden_size, output_dim)
        self.ode_unfolds = ode_unfolds
        self.tau = tau
        self.dt = dt

        self.cell = LTCCell(
            input_size=input_dim,
            hidden_size=hidden_size,
            tau=tau,
            dt=dt,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            out: (batch, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        h = torch.zeros(batch_size, self.hidden_size, device=device)

        for t in range(seq_len):
            for _ in range(self.ode_unfolds):
                h = self.cell(x[:, t, :], h)

        h = self.dropout(h)
        return self.fc(h)              # (batch, output_dim)

    def model_summary(self) -> dict:
        base = super().model_summary()
        base.update({
            "tau": self.tau,
            "dt": self.dt,
            "ode_unfolds": self.ode_unfolds,
        })
        return base


def build_lnn(config: dict, input_dim: int) -> LiquidNN:
    """
    Factory function — builds LiquidNN from config dict.

    Expected config keys:
        models.hidden_size
        models.dropout
        models.lnn.tau_constant
        models.lnn.ode_unfolds
        models.lnn.dt
        training.forecast_horizon
    """
    lnn_cfg = config["models"]["lnn"]
    return LiquidNN(
        input_dim=input_dim,
        hidden_size=config["models"]["hidden_size"],
        output_dim=config["training"]["forecast_horizon"],
        tau=lnn_cfg["tau_constant"],
        dt=lnn_cfg["dt"],
        ode_unfolds=lnn_cfg["ode_unfolds"],
        dropout=config["models"]["dropout"],
    )
