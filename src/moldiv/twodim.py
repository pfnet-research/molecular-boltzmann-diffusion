"""
Minimal tutorial
"""

import math

import numpy as np
from scipy.stats import qmc
import torch
import torch.nn as nn
from torch.func import jacrev, vmap, jacfwd
from torch import Tensor
from tqdm.auto import tqdm

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions

from moldiv.util import (
    ResNetModule,
    get_alpha_i,
    get_beta_i,
    get_sigma_i,
)


def log_p_dist(x: Tensor, sigma: float = 0.2):
    """
    Args:
        x (Tensor): Positions

    Returns:
        Tensor: log sum gauss mixture

    """
    device = x.device
    dist0 = torch.distributions.MultivariateNormal(
        torch.tensor([-2, -2], dtype=torch.float).to(device),
        sigma * torch.eye(2, dtype=torch.float).to(device),
    )
    lp0 = dist0.log_prob(x)
    dist1 = torch.distributions.MultivariateNormal(
        torch.tensor([2, 2], dtype=torch.float).to(device),
        sigma * torch.eye(2, dtype=torch.float).to(device),
    )
    lp1 = dist1.log_prob(x)
    dist2 = torch.distributions.MultivariateNormal(
        torch.tensor([0, 0], dtype=torch.float).to(device),
        (sigma / 2) * torch.eye(2, dtype=torch.float).to(device),
    )
    lp2 = dist2.log_prob(x)
    return torch.logsumexp(
        torch.vstack((lp0 + math.log(4), lp1 + math.log(4), lp2)).transpose(0, 1),
        dim=-1,
    ) - math.log(9)


def egrad(x, func: str = "triple-well"):
    """
    Calculate energy gradient

    Args:
        x (Tensor): Positions
        func (str): "triple-well" or "morse"

    Returns:
        Tensor: Gradient
    """
    x = x.clone().requires_grad_()
    match func:
        case "triple-well":
            lp = log_p_dist(x)
        case "morse":
            lp = log_p_morse(x)
        case _:
            raise ValueError(f"{func=} is invalid")
    return torch.autograd.grad(lp, x, grad_outputs=torch.ones_like(lp))[0]


def log1mexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    mask = -math.log(2) > x  # x < 0
    return torch.where(
        mask,
        (-x.exp()).log1p(),
        (-x.expm1()).log(),
    )


def log_p_morse(x: Tensor, beta=10, De=1.0, a=2.0, r_e=2.0):
    """
    q = exp(-βV)

    where

    V = De(1-exp(-a(r-r_e)))^2

    Therefore, log q = -βV
    """
    assert x.ndim == 2
    r = torch.norm(x, dim=1)
    return -beta * De * (-a * (r - r_e)).expm1().pow(2)


def sample_data(
    n_sample: int = 1024,
    scale: float = 4.0,
    func: str = "triple-well",
) -> tuple[Tensor, Tensor]:
    """
    Generate dataset by quasi Monte-Carlo

    Args:
        n_sample (int): # of sample data
        scale (float): position scale
        func (str): "triple-well" or "morse"

    Returns:
        tuple[Tensor, Tensor]: position, score
    """
    dim = 2
    engine = qmc.Sobol(dim)
    x_01 = engine.random(n_sample)
    x = 2 * scale * torch.tensor(x_01) - scale
    s = egrad(x, func)
    return x.to(torch.float32), s.to(torch.float32)


class PIDPDataset(torch.utils.data.TensorDataset):
    def __init__(self, x0: Tensor, s0: Tensor, beta_i: Tensor, t_batch: int):
        self.x0 = x0
        self.s0 = s0
        assert x0.shape == s0.shape
        self.beta_i = beta_i
        self.alpha_i = get_alpha_i(beta_i)
        self.sigma_i = get_sigma_i(self.alpha_i)
        self.N = len(beta_i) - 1
        assert 0 < t_batch <= self.N
        self.t_batch = t_batch
        self.M = len(x0)
        self.indices = np.arange(1, self.N + 1)
        self.h = 1 / self.N

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        """
        Returns:
            tuple: x0[idx], s0[idx], time, x_i[x0[idx]], beta_t
        """
        x0 = self.x0[idx]
        s0 = self.s0[idx]
        ϵ = torch.randn_like(x0)
        index = np.random.choice(self.indices, self.t_batch, replace=False)
        time = torch.tensor(index * self.h).to(torch.float32)

        β_t = self.beta_i[index] / self.h
        σ_i = self.sigma_i[index]
        α_i = self.alpha_i[index]

        x_rep = x0.repeat(self.t_batch, 1)
        ϵ = ϵ.repeat(self.t_batch, 1)

        assert isinstance(idx, int), f"{idx=}"

        x_i = torch.einsum("i,ij->ij", α_i, x_rep) + torch.einsum("i,ij->ij", σ_i, ϵ)
        return x0, s0, time, x_i, β_t


def get_dataloader(
    x0: Tensor,
    s0: Tensor,
    beta_i: Tensor,
    use_cuda: bool = True,
    batch_size: int = 128,
    batch_size_time: int = 32,
) -> torch.utils.data.DataLoader:
    dataset = PIDPDataset(x0=x0, s0=s0, beta_i=beta_i, t_batch=batch_size_time)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )
    return train_loader


def time_descriptor(t: Tensor, n: int) -> Tensor:
    """
    Args:
       t (Tensor): time input tensor with shape (D,)
       n (int): number of descriptor

    Returns:
       Tensor: time_descriptors with shape (n, D)
    """
    assert t.ndim == 1
    t = torch.unsqueeze(t, 1)
    emb_t0 = torch.cat(
        [
            t,
            torch.exp(-t * 3),
            torch.exp(-t * 12),
            torch.sin(torch.exp(-t * 6) * torch.tensor(math.pi)),  # math.piはダメ
            torch.exp(-t * 6),
        ],
        dim=1,
    )
    if n <= 5:
        return emb_t0[:, 0:n]
    else:
        raise NotImplementedError


class ScoreModel(nn.Module):
    """
    Score model with input (position + time), output (position)

    Args:
        mean (float): Mean for input standardization
        std (float): Standard deviation for input standrdization
        D (int): degree of freedoms (DOF) of positon
        N (int): number of discrete step
        n_time_desc (int, optional): number of time descriptors
        width_scale (int): The width scale of hidden layer
        n_resnet (int): number of ResNet Layer
        output_scale (float): Scale factor for output

    This docstring is generated by ChatGPT-4.
    """

    def __init__(
        self,
        *,
        input_mean=0,
        input_std=1,
        D=2,
        N=500,
        n_time_desc=5,
        width_scale=10,
        n_resnet=8,
        output_scale=8.0,
    ):
        super(ScoreModel, self).__init__()

        self.input_mean = input_mean
        self.input_std = input_std
        self.D = D
        self.N = N
        self.n_time_desc = n_time_desc
        self.output_scale = output_scale
        self.linears = nn.ModuleList(
            [
                nn.Linear(D + n_time_desc, width_scale * (D + n_time_desc)),
            ]
            + [ResNetModule(width_scale * (D + n_time_desc)) for _ in range(n_resnet)]
            + [
                nn.Linear(width_scale * (D + n_time_desc), D),
            ]
        )

    def __call__(self, x: Tensor, t: Tensor) -> tuple[Tensor, dict[int, Tensor]]:
        """
        モデルの呼び出し演算を行います。

        Args:
            x (Tensor): 入力データ
            t (Tensor): 入力時刻

        Returns:
            tuple: 順伝播の出力と各レイヤーでの特徴

        This docstring is generated by ChatGPT-4.
        """
        return super().__call__(x=x, t=t)

    def forward(self, x, t):
        """
        モデルの順伝播を計算します。

        Args:
            x (Tensor): position with shape (M, D)
            t (Tensor): time with shape (M,) must be in [0, 1]

        Returns:
            tuple: 順伝播の出力と各レイヤーでの特徴

        This docstring is generated by ChatGPT-4.
        """
        # assert torch.min(t) >= 0.0, "time must be in [0.0, 1.0]"
        # assert torch.max(t) <= 1.0, "time must be in [0.0, 1.0]"
        t_desc = time_descriptor(t, self.n_time_desc)
        x = (x - self.input_mean) / self.input_std
        # x = position_descriptor(x) <- layer also should be changed
        h = torch.cat((x, t_desc), dim=1)
        layer_features: dict[int, Tensor] = {}
        for i, linear in enumerate(self.linears):
            h = linear(h)
            layer_features[i] = h
        return h * self.output_scale, layer_features

    def div(
        self, x: Tensor, t: Tensor, use_hutchinson: bool = False, n_hutchinson: int = 20
    ) -> Tensor:
        """
        Evaluate divergence ∇⋅s
        """

        def func_vmap(x: Tensor, t: Tensor) -> Tensor:
            return self(x.unsqueeze(0), t.unsqueeze(0))[0].squeeze(0)

        if use_hutchinson:
            raise NotImplementedError
        else:
            ds_dx = vmap(jacrev(func_vmap, argnums=0))(x, t)
            ds_dx_trace = torch.einsum("...ii->...", ds_dx)
        return ds_dx_trace


def get_extension_manager(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    out_dir: int = "result",
    device: str = "cuda",
    snapshot: int | None = None,
) -> ppe.training.ExtensionsManager:
    my_extensions = [
        extensions.LogReport(
            trigger=(
                lambda EMP: (
                    EMP.iteration % EMP._iters_per_epoch == 0 or EMP.iteration == 1
                )
            ),
        ),
        extensions.ProgressBar(),
        extensions.ParameterStatistics(model, prefix="model"),
        extensions.Evaluator(
            valid_loader,
            model,
            eval_func=lambda x0, s0, t, xt, bt: _test(
                model,
                x0,
                s0,
                t,
                xt,
                bt,
                device=device,
            ),
        ),
        extensions.VariableStatisticsPlot(model),
        extensions.PlotReport(["train/loss", "val/loss"], "epoch", filename="loss.png"),
        extensions.PrintReport(
            [
                "iteration",
                "train/loss",
                "val/loss",
                "val/loss_t1",
                "val/loss_t0",
                "val/loss_fp",  # Fokker-Planck
                "lr",
                "elapsed_time",
            ]
        ),
        extensions.snapshot(autoload=True),
        ppe.training.ExtensionEntry(
            extensions.LRScheduler(lr_scheduler),
            trigger=(1, "epoch"),
        ),
    ]

    manager = ppe.training.ExtensionsManager(
        model,
        optimizer,
        epochs,
        extensions=my_extensions,
        iters_per_epoch=len(train_loader),
        stop_trigger=None,
        out_dir=out_dir,
    )
    # load the snapshot
    if snapshot is not None:
        state = torch.load(snapshot)
        manager.load_state_dict(state)
    return manager


def get_loss_t0(
    model: torch.nn.Module,
    x: Tensor,
    s: Tensor,
    weight: float,
    loss_func=torch.nn.MSELoss(reduction="mean"),
):
    """
    Args:
       model (torch.nn.Module): model which predict score from (x,t)
       x (Tensor): train position
       s (Tensor): train score
       weight (float): weight of this loss function
    """
    assert x.ndim == 2
    assert x.shape == s.shape
    t = torch.zeros(x.shape[0], device=x.device)
    s_hat, _ = model(x, t)
    return weight * loss_func(s, s_hat)


def get_loss_t1(
    model: torch.nn.Module,
    x: Tensor,
    weight: float,
    loss_func=torch.nn.MSELoss(reduction="mean"),
    mean: float = 0.0,
    std: float = 1.0,
):
    """
    Args:
       model (torch.nn.Module): model which predict score from (x,t)
       x (Tensor): train position
       mean (float): mean of p(t=1)
       std (float): standard deviation of p(t=1)
       weight (float): weight of this loss function
    """
    assert x.ndim == 2
    t = torch.ones(x.shape[0], device=x.device)
    s_hat, _ = model(x, t)
    return weight * loss_func(s_hat, -(x - mean) / std)


def get_loss_fokker_planck(
    model: ScoreModel,
    t: Tensor,
    x: Tensor,
    beta_t: Tensor,
    weight: float,
    use_hutchinson: bool = False,
    n_hutchinson: int = 20,  # <- DiG Table C2
    loss_func=torch.nn.HuberLoss(delta=0.1, reduction="mean"),
):
    assert x.shape[0] == t.shape[0] == beta_t.shape[0]
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    t = t.reshape(t.shape[0] * t.shape[1])
    beta_t = beta_t.reshape(beta_t.shape[0] * beta_t.shape[1])
    assert x.shape[0] == t.shape[0] == beta_t.shape[0]

    t.requires_grad_()
    x.requires_grad_()

    def func_vmap(x: Tensor, t: Tensor) -> Tensor:
        return model(x.unsqueeze(0), t.unsqueeze(0))[0].squeeze(0)

    # with forward_ad.dual_level():
    #     t_plus_dt = forward_ad.make_dual(t, torch.ones_like(t))
    #     (s_plus_ds_dt, features) = model(x, t_plus_dt)
    #     (s, ds_dt) = forward_ad.unpack_dual(s_plus_ds_dt)
    # DEPRECATED---
    # This does not work.
    s, _ = model(x, t)
    ds_dt = vmap(jacfwd(func_vmap, argnums=1))(x, t)
    # -----

    ds_dx_trace = model.div(x, t, use_hutchinson, n_hutchinson)
    mean = model.input_mean
    std = model.input_std
    xs_plus_ss = torch.sum(s * (s + (x - mean) / std), dim=-1)
    for_nabla = xs_plus_ss + ds_dx_trace
    (nabla_for_nabla,) = torch.autograd.grad(
        for_nabla,
        x,
        grad_outputs=torch.ones_like(for_nabla),
        create_graph=True,
    )
    norm = torch.sqrt(
        torch.square(
            torch.einsum("ji,j->ji", nabla_for_nabla, beta_t / 2) - ds_dt
        ).mean(dim=-1)
    )
    return weight * loss_func(norm, torch.zeros_like(t))


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    manager: ppe.training.ExtensionsManager,
    device: str = "cuda",
):
    while not manager.stop_trigger:
        model.train()
        for iteration, (x0, s0, t, xt, βt) in enumerate(train_loader):
            with manager.run_iteration():
                x0 = x0.to(device)
                s0 = s0.to(device)
                t = t.to(device)
                xt = xt.to(device)
                βt = βt.to(device)
                optimizer.zero_grad()
                loss_t0 = get_loss_t0(model, x0, s0, weight=1.0)
                loss_t1 = get_loss_t1(model, x0, weight=1.0)
                loss_fp = get_loss_fokker_planck(model, t, xt, βt, weight=1.0)
                loss = loss_t0 + loss_t1 + loss_fp
                ppe.reporting.report({"train/loss": loss.item()})
                loss.backward()
                ppe.reporting.report({"lr": optimizer.param_groups[0]["lr"]})
                optimizer.step()


def _test(
    model,
    x0: Tensor,
    s0: Tensor,
    t: Tensor,
    xt: Tensor,
    beta_t: Tensor,
    device: str = "cuda",
):
    x0 = x0.to(device)
    s0 = s0.to(device)
    t = t.to(device)
    xt = xt.to(device)
    beta_t = beta_t.to(device)
    loss_t0 = get_loss_t0(model, x0, s0, weight=1.0)
    loss_t1 = get_loss_t1(model, x0, weight=1.0)
    with torch.enable_grad():
        loss_fp = get_loss_fokker_planck(model, t, xt, beta_t, weight=1.0)
    loss = loss_t0 + loss_t1 + loss_fp
    # if lr_scheduler is not None:
    #    lr_scheduler.step(loss.item())
    ppe.reporting.report({"val/loss_t0": loss_t0.item()})
    ppe.reporting.report({"val/loss_t1": loss_t1.item()})
    ppe.reporting.report({"val/loss_fp": loss_fp.item()})
    ppe.reporting.report({"val/loss": loss.item()})


def sample_x_and_logP(
    n_sample: int, model: ScoreModel, device: str = "cuda:0"
) -> tuple[Tensor, Tensor]:
    """
    Sample x0 and logP(x0)

    log p(x0) = log N(x1) - ∑β_t/2 ∇s(xt, t)dt - D/2 ∑ β_t dt
    """
    from torch.distributions import MultivariateNormal

    beta_i = get_beta_i(model.N)
    h = 1 / model.N
    beta_t = beta_i / h
    x1 = torch.randn(n_sample, model.D, device=device)
    x_traj = ode_flow_traj(model, x1, device)
    x0 = x_traj[0]
    D = model.D

    log_p_simple = MultivariateNormal(
        torch.zeros_like(x1), torch.eye(D).to(device)
    ).log_prob(x1)

    ints_div_score = eval_ints_div_score(model, x_traj, beta_t, device)

    ints_sum_beta = torch.sum(beta_t) * h

    return x0, log_p_simple - ints_div_score - D / 2 * ints_sum_beta


def ode_flow_traj(model: torch.nn.Module, x1: Tensor, device: str = "cuda:0") -> Tensor:
    """
    Sampling by Ordinary differential equation (ODE)

    dR_t = -β_t/2 (R_t + s(R_t, t)) dt

    """
    N = model.N
    D = model.D
    dt = -1.0 / N
    h = 1.0 / N
    x_i = x1
    n_sample = len(x1)
    t_ones = torch.ones(n_sample, device=device)
    i = torch.arange(1, N + 1)
    β = get_beta_i(N)
    x_traj = torch.zeros((N, n_sample, D), dtype=torch.float32, device=device)
    x_traj[N - 1] = x1
    for i in tqdm(range(N, 0, -1)):
        t = i * h
        β_i = β[i]
        time = t * t_ones
        score, _ = model(x_i, time)
        β_t = β_i / h
        x_i = x_i - (x_i + score) * β_t / 2.0 * dt  # Intentionally overwitten
        x_traj[i - 2] = x_i
    return x_traj


def eval_ints_div_score(
    model: ScoreModel, x_traj: Tensor, beta_t: Tensor, device: str = "cuda:0"
) -> Tensor:
    """
    ∫ β_t/2∇⋅s dt
    """
    N = model.N
    dt = 1 / N
    n_sample = x_traj.shape[1]
    t_ones = torch.ones(n_sample, device=device)
    sum_ = torch.zeros(n_sample, device=device)
    for i in range(N):
        div_s = model.div(x_traj[i], (i + 1) * dt * t_ones)
        sum_ += beta_t[i] / 2.0 * div_s
    return sum_ * dt
