from torch import nn
import torch.nn.functional as F
import torch
from torch import Tensor

from loguru import logger


class ResNetModule(nn.Module):
    """
    ResNet
    """

    def __init__(
        self,
        inp_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = inp_dim
        if out_dim is None:
            out_dim = inp_dim
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(inp_dim)
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return residual + x


def set_device_around(use_cuda: bool, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"{use_cuda=}, {device=}")
    return device


def get_beta_i(
    N: int,
    beta_start: float = 1.0e-07,
    beta_end: float = 2.0e-02,  # N*beta_end=1.0 works well...
) -> Tensor:
    """
    construct beta_i

    Args:
        N: タイムステップ総数
        beta_start: βの初期値
        beta_end: βの最終値

    Returns:
        β_iの値 (i=0,1,...,N)

    """
    i = torch.arange(0, N + 1)
    return (
        1.0 / (1.0 + torch.exp(12.0 * (0.5 - i / N))) * (beta_end - beta_start)
        + beta_start
    )


def get_alpha_i(beta_i: Tensor) -> Tensor:
    """
    αᵢ = ∏_j^i √(1-βⱼ)
    """
    assert beta_i.ndim == 1, f"{beta_i.shape=} {beta_i.ndim=}"
    assert torch.max(beta_i) <= 1.0
    assert torch.min(beta_i) >= 0.0
    return torch.cumprod(torch.sqrt(1.0 - beta_i), dim=0)


def get_sigma_i(alpha_i: Tensor) -> Tensor:
    """
    σᵢ = √(1-αᵢ²)
    """
    assert alpha_i.ndim == 1, f"{alpha_i.shape=}"
    assert torch.max(alpha_i) <= 1.0
    assert torch.min(alpha_i) >= 0.0
    return torch.sqrt(1 - alpha_i**2)


def get_x_i(
    i: int,
    x0: Tensor,
    alpha_i: Tensor,
    sigma_i: Tensor,
) -> Tensor:
    """
    Rᵢ = αᵢR₀ + σᵢϵᵢ
    """
    return alpha_i[i] * x0 + sigma_i[i] * torch.randn_like(x0)
