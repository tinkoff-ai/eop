"""
Taken from here https://github.com/polixir/NeoRL/ and modified.
"""
import torch

from torch import nn as nn
from torch.distributions import Distribution, Normal
from typing import Tuple, Optional


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X = tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(
        self, 
        normal_mean: torch.Tensor,
        normal_std : torch.Tensor,
        epsilon    : float=1e-6
    ):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std  = normal_std
        self.normal      = Normal(normal_mean, normal_std)
        self.epsilon     = epsilon
        self.mode        = torch.tanh(normal_mean)

    def sample_n(
        self, 
        n                    : int,
        return_pre_tanh_value: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def atanh(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def log_prob(
        self, 
        value         : torch.Tensor,
        pre_tanh_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = self.atanh(value)

        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(
        self, 
        return_pretanh_value: bool = False
    ) -> torch.Tensor:
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
    
    def rsample(
        self, 
        return_pretanh_value: bool = False
    ) -> torch.Tensor:
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                torch.zeros(self.normal_mean.size(), device=self.normal_mean.device),
                torch.ones(self.normal_std.size(), device=self.normal_mean.device)
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)