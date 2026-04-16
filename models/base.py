# models/base.py
# Abstract base class for all sequence models in this pipeline.
#
# CONTRACT every subclass must honour:
#   Input  shape : (batch, seq_len, input_dim)
#   Output shape : (batch, output_dim)  — MIMO, all horizon steps at once
#
# Identity injection is handled HERE so no subclass repeats the logic.
# Each subclass calls  x = self._inject(x, identity)  at the top of forward()
# before touching its recurrent core.
#
# Why ABC + nn.Module?
#   ABC enforces that every architecture defines forward().
#   nn.Module gives parameters(), to(), state_dict() etc. for free.

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class BaseModel(ABC, nn.Module):
    """
    All stock prediction models must subclass this.

    Args:
        input_dim    : raw feature dimension BEFORE identity injection
        hidden_size  : recurrent / hidden layer width
        output_dim   : forecast horizon (number of future steps predicted)
        use_identity : whether to concatenate the Stock2Vec vector
        identity_dim : length of the Stock2Vec vector (ignored if not use_identity)
    """

    def __init__(self,
                 input_dim:    int,
                 hidden_size:  int,
                 output_dim:   int,
                 use_identity: bool = False,
                 identity_dim: int  = 4):
        super().__init__()
        self.input_dim    = input_dim
        self.hidden_size  = hidden_size
        self.output_dim   = output_dim
        self.use_identity = use_identity
        self.identity_dim = identity_dim

        # Effective input size seen by the recurrent core
        self.effective_input_dim = (
            input_dim + identity_dim if use_identity else input_dim
        )

    # ── Identity injection helper ─────────────────────────────────
    def _inject(self,
                x:        torch.Tensor,
                identity: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Tile the identity vector across every time-step and concatenate.

        x        : (B, T, input_dim)
        identity : (B, identity_dim)  or None
        returns  : (B, T, effective_input_dim)
        """
        if self.use_identity and identity is not None:
            id_exp = identity.unsqueeze(1).expand(-1, x.size(1), -1)
            return torch.cat([x, id_exp], dim=-1)
        return x

    # ── Abstract interface ────────────────────────────────────────
    @abstractmethod
    def forward(self,
                x:        torch.Tensor,
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x        : (batch, seq_len, input_dim)
            identity : (batch, identity_dim)  — None for baseline models
        Returns:
            (batch, output_dim)
        """

    # ── Shared utilities ──────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> dict:
        return {
            "architecture":     self.__class__.__name__,
            "input_dim":        self.input_dim,
            "effective_input":  self.effective_input_dim,
            "hidden_size":      self.hidden_size,
            "output_dim":       self.output_dim,
            "use_identity":     self.use_identity,
            "identity_dim":     self.identity_dim if self.use_identity else 0,
            "trainable_params": self.count_parameters(),
        }

    def __repr__(self) -> str:
        m   = self.model_summary()
        tag = "Identity" if self.use_identity else "Baseline"
        return (f"{m['architecture']}[{tag}]  "
                f"horizon={self.output_dim}  "
                f"params={m['trainable_params']:,}")
