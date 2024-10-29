import torch
from torch import Tensor

import numpy as np

from moldiv.util import get_alpha_i, get_sigma_i
from moldiv.molecule.loss import (
    get_loss_data,
    get_loss_t0,
    get_loss_t1,
    get_loss_fokker_planck,
)
import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions

from moldiv.molecule.transformer import ScoreTransformer


def sample_free_com(x0: Tensor) -> Tensor:
    """
    Sample ϵ from center of mass free Normal(0,1)

    Args:
        x0 (Tensor): tensor which has the same shape as noise

    Returns:
        Tensor: noise with shape (n_atom, 3)

    """
    assert x0.ndim == 2
    n_atom = len(x0)
    dim = 3
    mu = torch.zeros(n_atom * dim)
    A = torch.eye(len(mu)).reshape(n_atom, dim, n_atom, dim)
    for i in range(dim):
        A[:, i, :, i] -= 1 / n_atom * torch.ones(n_atom, n_atom)
    A = A.reshape(n_atom * dim, n_atom * dim)
    dist = torch.distributions.MultivariateNormal(
        mu, torch.eye(n_atom * dim, n_atom * dim)
    )
    ϵ = dist.sample()
    return (torch.einsum("ij,i->j", A, ϵ)).reshape(n_atom, dim)


def sample_free_com_batch(x0: Tensor) -> Tensor:
    """
    Sample ϵ (batch) from center of mass free Noraml(0,1)

    Args:
        x0 (Tensor): tensor which has the same shape as noise

    Returns:
        Tensor: noise with shape (n_batch, n_atom, 3)

    """
    assert x0.ndim == 3
    n_batch, n_atom, dim = x0.shape
    mu = torch.zeros(n_atom * dim)
    A = torch.eye(len(mu)).reshape(n_atom, dim, n_atom, dim)
    for i in range(dim):
        A[:, i, :, i] -= 1 / n_atom * torch.ones(n_atom, n_atom)
    A = A.reshape(n_atom * dim, n_atom * dim)
    dist = torch.distributions.MultivariateNormal(
        mu, torch.eye(n_atom * dim, n_atom * dim)
    )
    ϵ = dist.sample((n_batch,))
    return (torch.einsum("ij,ni->nj", A, ϵ)).reshape(n_batch, n_atom, dim)


class PIDPDataset(torch.utils.data.TensorDataset):
    """
    Physics-Informed Diffusion Pre-training (PIDP) dataset

    Args:
        x0 (Tensor): Positions
        Z (Tensor): Atomic Numbers
        s0 (Tensor): Score s(x0, t=0)
        beta_i (Tensor): Time steps
        t_batch (int): batch size of time
        ub_time (float): upper bound of time in [0.0, 1.0]
        always_include_t0_fp (bool): Whether or not always include FP loss at t=0
        tau (float): maxtime.

    """

    def __init__(
        self,
        x0: Tensor,
        Z: Tensor,
        s0: Tensor,
        beta_i: Tensor,
        t_batch: int,
        ub_time: float = 1.0,
        always_include_t0_fp: bool = True,
        tau: float = 1.0,
    ):
        self.x0 = x0
        self.Z = Z
        self.s0 = s0
        assert x0.shape == s0.shape
        assert x0.shape[0:2] == Z.shape[0:2]
        self.beta_i = beta_i
        self.alpha_i = get_alpha_i(beta_i)
        self.sigma_i = get_sigma_i(self.alpha_i)
        self.N = len(beta_i) - 1  # Size of discrete time steps
        assert 0 < t_batch <= self.N
        self.t_batch = t_batch
        self.M = len(x0)  # Size of positions
        self.h = tau / self.N
        self.ub_time = ub_time
        self.always_include_t0_fp = always_include_t0_fp
        self.tau = tau
        self.rng = np.random.default_rng()
        if ub_time != 1.0:
            self.indices = np.arange(1, min(self.N, int(ub_time / self.h)) + 1)
        else:
            self.indices = np.arange(1, self.N + 1)
        if always_include_t0_fp:
            p = np.exp(-np.linspace(0.0, 2.0, len(self.indices[1:])))
        else:
            p = np.exp(-np.linspace(0.0, 2.0, len(self.indices)))
        self.p = p / np.sum(p, dtype=float)

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        """
        Returns:
            tuple: x0[idx], Z[idx], s0[idx], time, x_i[x0[idx]], Z[idx], beta_t
        """
        x0 = self.x0[idx]
        x0 -= center_of_mass(x0)
        Z = self.Z[idx]
        s0 = self.s0[idx]
        # ϵ = torch.randn_like(x0)
        ϵ = sample_free_com(x0)
        if self.always_include_t0_fp:
            index = self.rng.choice(
                self.indices[1:], self.t_batch - 1, replace=False, p=self.p
            )
            index = np.append(index, 1)
        else:
            index = self.rng.choice(self.indices, self.t_batch, replace=False, p=self.p)
        time = torch.tensor(index * self.h).to(torch.float32)

        β_t = self.beta_i[index] / self.h
        σ_i = self.sigma_i[index]
        α_i = self.alpha_i[index]

        x_rep = x0.repeat(self.t_batch, 1, 1)
        Z_rep = Z.repeat(self.t_batch, 1)
        ϵ = ϵ.repeat(self.t_batch, 1, 1)
        # logger.debug(f"{x_rep.shape=}")
        # logger.debug(f"{Z_rep.shape=}")

        assert isinstance(idx, int), f"{idx=}"

        x_i = torch.einsum("i,ijk->ijk", α_i, x_rep) + torch.einsum(
            "i,ijk->ijk", σ_i, ϵ
        )
        return x0, Z, s0, time, x_i, Z_rep, β_t


class DenoiseDataset(torch.utils.data.TensorDataset):
    """
    Dataset for Train with data

    Args:
        x0 (Tensor): Positions
        Z (Tensor): Atomic Numbers
        s0 (Tensor): Score s(x0, t=0)
        beta_i (Tensor): Time steps
        noise_batch (int): batch size of noise
        ub_time (float): upper bound of time in [0.0, 1.0]
        tau (float): maxtime.

    """

    def __init__(
        self,
        x0,
        Z,
        beta_i,
        noise_batch,
        ub_time: float = 1.0,
        tau: float = 1.0,
    ):
        self.x0 = x0
        self.Z = Z
        assert x0.shape[0:2] == Z.shape[0:2]
        self.beta_i = beta_i
        self.alpha_i = get_alpha_i(beta_i)
        self.sigma_i = get_sigma_i(self.alpha_i)
        self.N = len(beta_i) - 1
        self.M = len(x0)
        self.h = tau / self.N
        self.noise_batch = noise_batch
        if ub_time != 1.0:
            self.indices = np.arange(1, min(self.N, int(ub_time / self.h)) + 1)
        else:
            self.indices = np.arange(1, self.N + 1)

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        """
        Returns:
            tuple: time, x_i[x0[idx]], Z[idx], sigma_i, epsilon_i
        """
        x0 = self.x0[idx]  # (n_atom, 3)
        x0 -= center_of_mass(x0)
        Z = self.Z[idx]  # (n_atom,)
        x_rep = x0.repeat(self.noise_batch, 1, 1)  # (noi_batch, n_atom, 3)
        Z_rep = Z.repeat(self.noise_batch, 1)  # (noi_batch, n_atom)
        ϵ_i = sample_free_com_batch(x_rep)  # (noi_batch, n_atom, 3)

        index = np.random.choice(self.indices, 1, replace=False)
        time = torch.tensor(index * self.h).to(torch.float32).repeat(self.noise_batch)
        # (noi_batch, n_atom)

        σ_i = self.sigma_i[index]  # int
        α_i = self.alpha_i[index]  # int

        x_i = α_i * x_rep + σ_i * ϵ_i

        assert isinstance(idx, int), f"{idx=}"

        return time, x_i, Z_rep, σ_i, ϵ_i


class PIandDenoiseDataset(torch.utils.data.TensorDataset):
    """
    Physics-Informed (PI) & Denoising score matching dataset

    Args:
        x0 (Tensor): Positions for PI
        Z (Tensor): Atomic Numbers for PI
        s0 (Tensor): Score s(x0, t=0) for PI
        x_data (Tensor): Positions for Denoising
        Z_data (Tensor): Atomic numbers for Denoising
        beta_i (Tensor): Time steps
        t_batch (int): batch size of time
        noise_batch (int): batch size of noise
        ub_time (float): upper bound of time in [0.0, 1.0]
        always_include_t0_fp (bool): Whether or not always include FP loss at t=0
        tau (float): maxtime.

    """

    def __init__(
        self,
        x0: Tensor,
        Z: Tensor,
        s0: Tensor,
        x_data: Tensor,
        Z_data: Tensor,
        beta_i: Tensor,
        t_batch: int = 1,
        noise_batch: int = 32,
        ub_time: float = 1.0,
        always_include_t0_fp: bool = True,
        tau: float = 1.0,
    ):
        self.x0 = x0
        self.Z = Z
        self.s0 = s0
        self.x_data = x_data
        self.Z_data = Z_data
        assert x0.shape == s0.shape
        assert x0.shape[0:2] == Z.shape[0:2]
        self.beta_i = beta_i
        self.alpha_i = get_alpha_i(beta_i)
        self.sigma_i = get_sigma_i(self.alpha_i)
        self.N = len(beta_i) - 1
        assert 0 < t_batch <= self.N
        self.t_batch = t_batch
        self.noise_batch = noise_batch
        self.M = len(x0)
        self.h = tau / self.N
        self.ub_time = ub_time
        self.always_include_t0_fp = always_include_t0_fp
        self.tau = tau
        if ub_time != 1.0:
            self.indices = np.arange(1, min(self.N, int(ub_time / self.h)) + 1)
        else:
            self.indices = np.arange(1, self.N + 1)
        if always_include_t0_fp:
            p = np.exp(-np.linspace(0.0, 12.0, len(self.indices[1:])))
        else:
            p = np.exp(-np.linspace(0.0, 12.0, len(self.indices)))
        self.p = p / np.sum(p, dtype=float)

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        """
        Returns:
            tuple: x0[idx], Z[idx], s0[idx], time, x_i[x0[idx]], Z[idx], beta_t, time_de, x_i_de, Z_de, sigma_i, epsilon_i
        """
        x0 = self.x0[idx]
        x0 -= center_of_mass(x0)
        Z = self.Z[idx]
        s0 = self.s0[idx]
        # ϵ = torch.randn_like(x0)
        ϵ = sample_free_com(x0)
        if self.always_include_t0_fp:
            index = np.random.choice(
                self.indices[1:], self.t_batch - 1, replace=False, p=self.p
            )
            index = np.append(index, 1)
        else:
            index = np.random.choice(
                self.indices, self.t_batch, replace=False, p=self.p
            )
        time_pi = torch.tensor(index * self.h).to(torch.float32)

        β_t = self.beta_i[index] / self.h
        σ_i = self.sigma_i[index]
        α_i = self.alpha_i[index]

        x_rep_pi = x0.repeat(self.t_batch, 1, 1)
        Z_rep_pi = Z.repeat(self.t_batch, 1)
        ϵ_pi = ϵ.repeat(self.t_batch, 1, 1)

        assert isinstance(idx, int), f"{idx=}"

        x_i_pi = torch.einsum("i,ijk->ijk", α_i, x_rep_pi) + torch.einsum(
            "i,ijk->ijk", σ_i, ϵ_pi
        )

        x_de = self.x_data[idx]
        Z_de = self.Z_data[idx]
        x_rep_de = x_de.repeat(self.noise_batch, 1, 1)  # (noi_batch, n_atom, 3)
        Z_rep_de = Z_de.repeat(self.noise_batch, 1)  # (noi_batch, n_atom)
        ϵ_i_de = sample_free_com_batch(x_rep_de)  # (noi_batch, n_atom, 3)

        # Always exclude t=0 from denoising score matching
        index = np.random.choice(self.indices[1:], 1, replace=False)
        time_de = (
            torch.tensor(index * self.h).to(torch.float32).repeat(self.noise_batch)
        )
        σ_i_de = self.sigma_i[index]
        α_i_de = self.alpha_i[index]
        # (noi_batch, n_atom)

        x_i_de = α_i_de * x_rep_de + σ_i_de * ϵ_i_de
        return (
            x0,
            Z,
            s0,
            time_pi,
            x_i_pi,
            Z_rep_pi,
            β_t,
            time_de,
            x_i_de,
            Z_rep_de,
            σ_i_de,
            ϵ_i_de,
        )


def center_of_mass(x: Tensor):
    """
    Calculate center of mass of x

    Args:
        x (Tensor): Positions with shape (..., n_atom, 3)

    """
    assert x.shape[-1] == 3
    return x.mean(dim=-2, keepdim=True)


def get_dataloader(
    x0: Tensor,
    Z: Tensor,
    s0: Tensor,
    beta_i: Tensor,
    use_cuda: bool = True,
    batch_size: int = 32,
    batch_size_time: int = 32,
    batch_size_noise: int = 32,
    ub_time: float = 1.0,
    always_include_t0_fp: bool = True,
    tau: float = 1.0,
    train_with_data: bool = False,
    physics_informed: bool = True,
    x_data: Tensor | None = None,
    Z_data: Tensor | None = None,
) -> torch.utils.data.DataLoader:
    """
    Get Dataloader for training / validation

    Args:
        x0 (Tensor): Positions for PI
        Z (Tensor): Atomic Numbers for PI
        s0 (Tensor): Score s(x0, t=0) for PI
        beta_i (Tensor): Time steps
        use_cuda (bool): Whether use CUDA
        bathc_size (int): Batch size of position
        bathc_size_time (int): Batch size of time
        bathc_size_noise (int): Batch size of noise
        ub_time (float): upper bound of time in [0.0, 1.0]
        always_include_t0_fp (bool): Whether or not always include FP loss at t=0
        tau (float): maxtime.
        train_with_data (bool): Whether train with data
        physics_informed (bool): Whether train by Physics-Informed (PI)
        x_data (Tensor): Positions for Denoising
        Z_data (Tensor): Atomic numbers for Denoising
    """
    if train_with_data and physics_informed:
        if x_data is None:
            x_data = x0
            Z_data = Z
        assert Z_data is not None
        dataset = PIandDenoiseDataset(
            x0=x0,
            Z=Z,
            s0=s0,
            x_data=x_data,
            Z_data=Z_data,
            beta_i=beta_i,
            noise_batch=batch_size_noise,
            t_batch=batch_size_time,
            ub_time=ub_time,
            always_include_t0_fp=False,
            tau=tau,
        )
    elif train_with_data and not physics_informed:
        dataset = DenoiseDataset(
            x0=x0,
            Z=Z,
            beta_i=beta_i,
            noise_batch=batch_size_noise,
        )
    elif not train_with_data and physics_informed:
        dataset = PIDPDataset(
            x0=x0,
            Z=Z,
            s0=s0,
            beta_i=beta_i,
            t_batch=batch_size_time,
            ub_time=ub_time,
            always_include_t0_fp=always_include_t0_fp,
            tau=tau,
        )
    else:
        raise ValueError

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )
    return train_loader


def get_extension_manager(
    model: ScoreTransformer,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    out_dir: str = "result",
    device: str = "cuda",
    snapshot: int | None = None,
    weight_t0: float = 1.0,
    weight_t1: float = 1.0,
    weight_fp: float = 1.0,
    weight_with_data: float = 1.0,
    train_with_data: bool = False,
    physics_informed: bool = True,
    finite_diff: bool = False,
    wo_drift: bool = False,
) -> ppe.training.ExtensionsManager:
    """
    Get Extension Manager of pytorch-pfn-extras (ppe)

    Args:
        model (ScoreTransformer): Training model
        optimizer (torch.optim.Optimizer): Optimizer like Adam
        train_loader (torch.utils.data.DataLoader): Dataloader for training
        valid_loader (torch.utils.data.DataLoader): Dataloader for validation
        epochs (int): Number of epochs
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler | None): lerning rate scheduler
        out_dir (str): Output directory
        device (str): Device of the model & data
        snapshot (int | None): Restart snapshot
        weight_t0 (float): loss weight for t=0 loss
        weight_t1 (float): loss weight for t=1 loss (optional)
        weight_fp (float): loss weight for FP equation
        weight_with_data (float): loss weight for train with data
        train_with_data (bool): Whether train with data
        physics_informed (bool): Whether train by Physics-Informed (PI)
        finite_diff (bool): Use finite difference in ds/dt instead of autodiff
        wo_drift (bool): Diffusion without drift term

    """
    if train_with_data and physics_informed:
        eval_func = (
            lambda x0,
            Z0,
            s0,
            t_pi,
            xt_pi,
            Zt_pi,
            bt_pi,
            t_de,
            xt_de,
            Zt_de,
            sigma_i_de,
            eps_i_de: _test_pi_with_data(
                model,
                x0,
                Z0,
                s0,
                t_pi,
                xt_pi,
                Zt_pi,
                bt_pi,
                t_de,
                xt_de,
                Zt_de,
                sigma_i_de,
                eps_i_de,
                device=device,
                weight_t0=weight_t0,
                weight_t1=weight_t1,
                weight_fp=weight_fp,
                weight_with_data=weight_with_data,
                finite_diff=finite_diff,
                wo_drift=wo_drift,
            )
        )
        reports = [
            "iteration",
            "train/loss",
            "val/loss",
            "val/loss_t1",
            "val/loss_t0",
            "val/loss_fp",  # Fokker-Planck
            "val/loss_with_data",
            "lr",
            "elapsed_time",
        ]
    elif train_with_data and not physics_informed:
        eval_func = lambda t, xt, Zt, sigma_i, eps_i: _test_with_data(
            model,
            t,
            xt,
            Zt,
            sigma_i,
            eps_i,
            device=device,
        )
        reports = [
            "iteration",
            "train/loss",
            "val/loss",
            "lr",
            "elapsed_time",
        ]
    elif not train_with_data and physics_informed:
        eval_func = lambda x0, Z0, s0, t, xt, Zt, bt: _test(
            model,
            x0,
            Z0,
            s0,
            t,
            xt,
            Zt,
            bt,
            device=device,
            weight_t0=weight_t0,
            weight_t1=weight_t1,
            weight_fp=weight_fp,
            finite_diff=finite_diff,
            wo_drift=wo_drift,
        )
        reports = [
            "iteration",
            "train/loss",
            "val/loss",
            "val/loss_t1",
            "val/loss_t0",
            "val/loss_fp",  # Fokker-Planck
            "lr",
            "elapsed_time",
        ]
    else:
        raise ValueError
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
            eval_func=eval_func,
        ),
        extensions.VariableStatisticsPlot(model),
        extensions.PlotReport(["train/loss", "val/loss"], "epoch", filename="loss.png"),
        extensions.PrintReport(reports),
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


def train(
    model: ScoreTransformer,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    manager: ppe.training.ExtensionsManager,
    device: str = "cuda",
    weight_t0: float = 1.0,
    weight_t1: float = 0.0,
    weight_fp: float = 1.0,
    finite_diff: bool = True,
    wo_drift: bool = False,
):
    """
    Physics-Informed Train

    Args:
        model (ScoreTransformer): Training model
        optimizer (torch.optim.Optimizer): Optimizer like Adam
        train_loader (torch.utils.data.DataLoader): Dataloader for training
        manager (ppe.training.ExtensionsManager): Extensino Maneger of PPE
        device (str): Device of the model & data
        weight_t0 (float): loss weight for t=0 loss
        weight_t1 (float): loss weight for t=1 loss (optional)
        weight_fp (float): loss weight for FP equation
        finite_diff (bool): Use finite difference in ds/dt instead of autodiff
        wo_drift (bool): Diffusion without drift term

    """
    while not manager.stop_trigger:
        model.train()
        for iteration, (x0, Z0, s0, t, xt, Zt, βt) in enumerate(train_loader):
            with manager.run_iteration():
                x0 = x0.to(device)
                Z0 = Z0.to(device)
                s0 = s0.to(device)
                t = t.to(device)
                xt = xt.to(device)
                Zt = Zt.to(device)
                βt = βt.to(device)
                optimizer.zero_grad()
                loss = 0.0
                if weight_t0 > 0.0:
                    loss_t0 = get_loss_t0(
                        model,
                        x0,
                        Z0,
                        s0,
                        weight=weight_t0,
                    )
                    loss += loss_t0
                if weight_t1 > 0.0:
                    x1 = sample_free_com_batch(x0).to(x0.device)
                    loss_t1 = get_loss_t1(
                        model,
                        x1,
                        Z0,
                        weight=weight_t1,
                    )
                    loss += loss_t1
                if weight_fp > 0.0:
                    loss_fp = get_loss_fokker_planck(
                        model,
                        t,
                        xt,
                        Zt,
                        βt,
                        weight=weight_fp,
                        finite_diff=finite_diff,
                        wo_drift=wo_drift,
                    )
                    loss += loss_fp
                ppe.reporting.report({"train/loss": loss.item()})
                loss.backward()
                ppe.reporting.report({"lr": optimizer.param_groups[0]["lr"]})
                optimizer.step()


def train_with_data(
    model: ScoreTransformer,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    manager: ppe.training.ExtensionsManager,
    device: str = "cuda",
):
    """
    Train with data

    Args:
        model (ScoreTransformer): Training model
        optimizer (torch.optim.Optimizer): Optimizer like Adam
        train_loader (torch.utils.data.DataLoader): Dataloader for training
        manager (ppe.training.ExtensionsManager): Extensino Maneger of PPE
        device (str): Device of the model & data

    """
    while not manager.stop_trigger:
        model.train()
        for iteration, (t, xt, Zt, sigma_i, eps_i) in enumerate(train_loader):
            with manager.run_iteration():
                optimizer.zero_grad()
                t = t.to(device)
                xt = xt.to(device)
                Zt = Zt.to(device)
                sigma_i = sigma_i.to(device)
                eps_i = eps_i.to(device)
                loss = get_loss_data(model, t, xt, Zt, sigma_i, eps_i)
                ppe.reporting.report({"train/loss": loss.item()})
                loss.backward()
                ppe.reporting.report({"lr": optimizer.param_groups[0]["lr"]})
                optimizer.step()


def train_pidp_with_data(
    model: ScoreTransformer,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    manager: ppe.training.ExtensionsManager,
    device: str = "cuda",
    weight_t0: float = 1.0,
    weight_t1: float = 0.0,
    weight_fp: float = 1.0,
    weight_with_data: float = 1.0,
    finite_diff: bool = True,
    wo_drift: bool = False,
):
    """
    Physics-Informed & Data Train

    Args:
        model (ScoreTransformer): Training model
        optimizer (torch.optim.Optimizer): Optimizer like Adam
        train_loader (torch.utils.data.DataLoader): Dataloader for training
        manager (ppe.training.ExtensionsManager): Extensino Maneger of PPE
        device (str): Device of the model & data
        weight_t0 (float): loss weight for t=0 loss
        weight_t1 (float): loss weight for t=1 loss (optional)
        weight_fp (float): loss weight for FP equation
        weight_with_data (float): loss weight for train with data
        finite_diff (bool): Use finite difference in ds/dt instead of autodiff
        wo_drift (bool): Diffusion without drift term

    """
    while not manager.stop_trigger:
        model.train()
        for iteration, (
            x0,
            Z0,
            s0,
            t_pi,
            xt_pi,
            Zt_pi,
            βt_pi,
            t_de,
            xt_de,
            Zt_de,
            sigma_i_de,
            eps_i_de,
        ) in enumerate(train_loader):
            with manager.run_iteration():
                x0 = x0.to(device)
                Z0 = Z0.to(device)
                s0 = s0.to(device)
                t_pi = t_pi.to(device)
                xt_pi = xt_pi.to(device)
                Zt_pi = Zt_pi.to(device)
                βt_pi = βt_pi.to(device)
                optimizer.zero_grad()
                loss = 0.0
                if weight_t0 > 0.0:
                    loss_t0 = get_loss_t0(model, x0, Z0, s0, weight=weight_t0)
                    loss += loss_t0
                if weight_t1 > 0.0:
                    x1 = sample_free_com_batch(x0).to(x0.device)
                    loss_t1 = get_loss_t1(model, x1, Z0, weight=weight_t1)
                    loss += loss_t1
                if weight_fp > 0.0:
                    loss_fp = get_loss_fokker_planck(
                        model,
                        t_pi,
                        xt_pi,
                        Zt_pi,
                        βt_pi,
                        weight=weight_fp,
                        finite_diff=finite_diff,
                        wo_drift=wo_drift,
                    )
                    loss += loss_fp
                optimizer.zero_grad()
                t_de = t_de.to(device)
                xt_de = xt_de.to(device)
                Zt_de = Zt_de.to(device)
                sigma_i_de = sigma_i_de.to(device)
                eps_i_de = eps_i_de.to(device)
                if weight_with_data > 0.0:
                    loss_with_data = get_loss_data(
                        model,
                        t_de,
                        xt_de,
                        Zt_de,
                        sigma_i_de,
                        eps_i_de,
                        weight=weight_with_data,
                    )
                    loss += loss_with_data
                ppe.reporting.report({"train/loss": loss.item()})
                loss.backward()
                ppe.reporting.report({"lr": optimizer.param_groups[0]["lr"]})
                optimizer.step()


def _test(
    model: ScoreTransformer,
    x0: Tensor,
    Z0: Tensor,
    s0: Tensor,
    t: Tensor,
    xt: Tensor,
    Zt: Tensor,
    beta_t: Tensor,
    device: str = "cuda",
    weight_t0: float = 1.0,
    weight_t1: float = 0.0,
    weight_fp: float = 1.0,
    finite_diff: bool = False,
    wo_drift: bool = False,
):
    """
    Do not use this function directory

    Args:
        model (ScoreTransformer): Training model
        x0 (Tensor): Positions for PI
        Z0 (Tensor): Atomic Numbers for PI
        s0 (Tensor): Score s(x0, t=0) for PI
        t (Tensor): Time
        xt (Tensor): Position at t
        Zt (Tensor): Atomic numbers of xt
        beta_t (Tensor): The beta_t
        device (str): Device of the model & data
        weight_t0 (float): loss weight for t=0 loss
        weight_t1 (float): loss weight for t=1 loss (optional)
        weight_fp (float): loss weight for FP equation
        finite_diff (bool): Use finite difference in ds/dt instead of autodiff
        wo_drift (bool): Diffusion without drift term

    """
    x0 = x0.to(device)
    Z0 = Z0.to(device)
    s0 = s0.to(device)
    t = t.to(device)
    xt = xt.to(device)
    Zt = Zt.to(device)
    beta_t = beta_t.to(device)
    loss_t0 = get_loss_t0(model, x0, Z0, s0, weight=weight_t0)
    x1 = sample_free_com_batch(x0).to(x0.device)
    loss_t1 = get_loss_t1(model, x1, Z0, weight=weight_t1)
    with torch.enable_grad():
        loss_fp = get_loss_fokker_planck(
            model,
            t,
            xt,
            Zt,
            beta_t,
            weight=weight_fp,
            finite_diff=finite_diff,
            wo_drift=wo_drift,
        )
    loss = loss_t0 + loss_t1 + loss_fp
    ppe.reporting.report({"val/loss_t0": loss_t0.item()})
    ppe.reporting.report({"val/loss_t1": loss_t1.item()})
    ppe.reporting.report({"val/loss_fp": loss_fp.item()})
    ppe.reporting.report({"val/loss": loss.item()})


def _test_with_data(
    model: ScoreTransformer,
    t: Tensor,
    xt: Tensor,
    Zt: Tensor,
    sigma_i: Tensor,
    eps_i: Tensor,
    device: str = "cuda",
):
    """
    Do not use this function directory

    Args:
        model (ScoreTransformer): Training model
        t (Tensor): time
        xt (Tensor): positions
        Zt (Tensor): Atomic numbers
        sigma_i (Tensor): The sigma
        eps_i (Tensor): noise
        device (str): Device

    """
    t = t.to(device)
    xt = xt.to(device)
    Zt = Zt.to(device)
    sigma_i = sigma_i.to(device)
    eps_i = eps_i.to(device)
    loss = get_loss_data(model, t, xt, Zt, sigma_i, eps_i)
    ppe.reporting.report({"val/loss": loss.item()})


def _test_pi_with_data(
    model: ScoreTransformer,
    x0: Tensor,
    Z0: Tensor,
    s0: Tensor,
    t_pi: Tensor,
    xt_pi: Tensor,
    Zt_pi: Tensor,
    beta_t_pi: Tensor,
    t_de: Tensor,
    xt_de: Tensor,
    Zt_de: Tensor,
    sigma_i_de: Tensor,
    eps_i_de: Tensor,
    device: str = "cuda",
    weight_t0: float = 1.0,
    weight_t1: float = 0.0,
    weight_fp: float = 1.0,
    weight_with_data: float = 1.0,
    finite_diff: bool = False,
    wo_drift: bool = False,
):
    """
    Do not use this function directory

    Args:
        model (ScoreTransformer): Training model
        x0 (Tensor): Positions for PI
        Z0 (Tensor): Atomic Numbers for PI
        s0 (Tensor): Score s(x0, t=0) for PI
        t_pi (Tensor): Time for PI
        xt_pi (Tensor): Position at t for PI
        Zt_pi (Tensor): Atomic numbers of xt for PI
        beta_t_pi (Tensor): The beta_t
        t_de (Tensor): time fot denoising
        xt_de (Tensor): positions for denoising
        Zt_de (Tensor): Atomic numbers for denoising
        sigma_i_de (Tensor): The sigma for denoising
        eps_i_de (Tensor): noise for denoising
        device (str): Device of the model & data
        weight_t0 (float): loss weight for t=0 loss
        weight_t1 (float): loss weight for t=1 loss (optional)
        weight_fp (float): loss weight for FP equation
        weight_with_data (float): loss weight for train with data
        finite_diff (bool): Use finite difference in ds/dt instead of autodiff
        wo_drift (bool): Diffusion without drift term

    """
    x0 = x0.to(device)
    Z0 = Z0.to(device)
    s0 = s0.to(device)
    t_pi = t_pi.to(device)
    xt_pi = xt_pi.to(device)
    Zt_pi = Zt_pi.to(device)
    beta_t_pi = beta_t_pi.to(device)
    loss_t0 = get_loss_t0(model, x0, Z0, s0, weight=weight_t0)
    x1 = sample_free_com_batch(x0).to(x0.device)
    loss_t1 = get_loss_t1(model, x1, Z0, weight=weight_t1)
    with torch.enable_grad():
        loss_fp = get_loss_fokker_planck(
            model,
            t_pi,
            xt_pi,
            Zt_pi,
            beta_t_pi,
            weight=weight_fp,
            finite_diff=finite_diff,
            wo_drift=wo_drift,
        )
    t_de = t_de.to(device)
    xt_de = xt_de.to(device)
    Zt_de = Zt_de.to(device)
    sigma_i_de = sigma_i_de.to(device)
    eps_i_de = eps_i_de.to(device)
    loss_with_data = get_loss_data(
        model, t_de, xt_de, Zt_de, sigma_i_de, eps_i_de, weight=weight_with_data
    )
    loss = loss_t0 + loss_t1 + loss_fp + loss_with_data
    ppe.reporting.report({"val/loss_t0": loss_t0.item()})
    ppe.reporting.report({"val/loss_t1": loss_t1.item()})
    ppe.reporting.report({"val/loss_fp": loss_fp.item()})
    ppe.reporting.report({"val/loss_with_data": loss_with_data.item()})
    ppe.reporting.report({"val/loss": loss.item()})
