import torch
import torch.nn as nn
from models.base import BaseModel


class RNNModel(BaseModel):
    """
    Vanilla RNN for time-series forecasting.

    Input:  (batch, seq_len, input_dim)
    Output: (batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(RNNModel, self).__init__(input_dim, hidden_size, output_dim)
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.rnn(x)          # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]           # last timestep: (batch, hidden_size)
        out = self.dropout(out)
        return self.fc(out)           # (batch, output_dim)

    def model_summary(self) -> dict:
        base = super().model_summary()
        base.update({"num_layers": self.num_layers})
        return base


def build_rnn(config: dict, input_dim: int) -> RNNModel:
    """
    Factory function — builds RNNModel from config dict.

    Expected config keys (under models):
        hidden_size, num_layers, dropout
    And forecast_horizon under training.
    """
    return RNNModel(
        input_dim=input_dim,
        hidden_size=config["models"]["hidden_size"],
        output_dim=config["training"]["forecast_horizon"],
        num_layers=config["models"]["num_layers"],
        dropout=config["models"]["dropout"],
    )
