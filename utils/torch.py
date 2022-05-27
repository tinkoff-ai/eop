import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_sync(targ_model: nn.Module, model: nn.Module, tau: float) -> None:
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)


def hard_sync(targ_model: nn.Module, model: nn.Module) -> None:
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.copy_(p.data)


def compute_huber_loss(
    y     : torch.Tensor,
    target: torch.Tensor,
    beta  : float = 1.0
) -> torch.Tensor:
    diff = target - y
    cond = diff.detach().abs() < beta
    return torch.where(cond, 0.5 * diff ** 2, beta * (diff.abs() - 0.5 * beta))


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x