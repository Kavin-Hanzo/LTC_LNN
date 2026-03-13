import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):
    """
    All models must:
      - Accept input of shape  (batch, seq_len, input_dim)
      - Return output of shape (batch, forecast_horizon)
    """

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor of shape (batch, output_dim)
        """
        pass

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> dict:
        return {
            "architecture": self.__class__.__name__,
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
            "output_dim": self.output_dim,
            "trainable_params": self.count_parameters(),
        }
