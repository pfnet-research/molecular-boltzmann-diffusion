from .transformer import ScoreTransformer

import torch
from torch import Tensor
from torch.autograd import forward_ad


def get_loss_t0(
    model: ScoreTransformer,
    x: Tensor,
    Z: Tensor,
    s: Tensor,
    weight: float = 1.0,
    loss_func=torch.nn.HuberLoss(delta=50.0, reduction="mean"),
):
    """
    Args:
       model (torch.nn.Module): model which predict score from (x,t)
       x (Tensor): train position
       Z (Tensor): atom number
       s (Tensor): train score
       weight (float): weight of this loss function
       loss_func: Loss function
    """
    if weight <= 0.0:
        return torch.tensor(0.0).to(x.device)
    assert x.ndim == 3
    assert x.shape[2] == 3
    assert Z.ndim == 2
    assert Z.shape[0] == x.shape[0]
    assert x.shape == s.shape
    t = torch.ones(x.shape[0], device=x.device) * model.tau / model.N
    s_hat, _ = model(x, Z, t)

    return weight * loss_func(s, s_hat)


def get_loss_t1(
    model: ScoreTransformer,
    x: Tensor,
    Z: Tensor,
    weight: float = 1.0,
    loss_func=torch.nn.MSELoss(reduction="mean"),
    wo_drift: bool = False,
):
    """
    Args:
       model (ScoreTransformer): model which predict score from (x,t)
       x (Tensor): train position
       Z (Tensor): atom number
       weight (float): weight of this loss function

    A = QQᵀ
    Σ = (AAᵀ) = QQᵀ(QQᵀ)ᵀ = QQᵀQQᵀ = QQᵀ=A
    Σ^-1 = A^-1 = (QQᵀ)^-1 = Qᵀ^-1 Q^-1 = QQ^T = A
    """
    if weight <= 0.0:
        return torch.tensor(0.0).to(x.device)
    assert x.ndim == 3
    assert x.shape[2] == 3
    assert Z.ndim == 2
    assert Z.shape[0] == x.shape[0]
    t = torch.ones(x.shape[0], device=x.device)
    n_atom = x.shape[1]
    dim = 3
    mu = torch.zeros(n_atom * dim)
    A = torch.eye(len(mu)).reshape(n_atom, dim, n_atom, dim).to(x.device)
    ones = torch.ones(n_atom, n_atom).to(x.device)
    for i in range(dim):
        A[:, i, :, i] -= 1 / n_atom * ones
    s_hat, _ = model(x, Z, t)
    if wo_drift:
        return weight * (s_hat * s_hat).sum(dim=-1).sum(dim=-1).mean(dim=-1)
    else:
        return weight * loss_func(s_hat, -torch.einsum("ijkl,nkl->nij", A, x))


def get_loss_fokker_planck(
    model: ScoreTransformer,
    t: Tensor,
    x: Tensor,
    Z: Tensor,
    beta_t: Tensor,
    weight: float = 1.0,
    use_hutchinson: bool = False,
    n_hutchinson: int = 20,  # <- DiG Table C2
    loss_func=torch.nn.HuberLoss(delta=0.1, reduction="mean"),
    finite_diff: bool = False,
    wo_drift: bool = False,
):
    """
    Get Fokker-Planck loss function

    Args:
        model (ScoreTransformer): model
        t (Tensor): time with shape (t_batch, posi_batch,)
        x (Tensor): position with shape (t_batch, posi_batch, n_atom, 3)
        Z (Tensor): atom number with shape (t_batch, posi_batch, n_atom,)
        beta_t (Tensor): beta_t with shape (t_batch, posi_batch)
        use_hitchison (bool): Use hutchinson estimation for divergence.
        n_hutchinson (int): Number of vector for hutchinson estimation
        loss_func (Callable): Loss function
        finite_diff (bool): Use finite differcen rather than auto diff for ds/dt.
        wo_unif (bool): without drift term
    """
    if weight <= 0.0:
        return torch.tensor(0.0).to(x.device)
    assert x.shape[0] == t.shape[0] == Z.shape[0] == beta_t.shape[0]
    assert x.ndim == 4
    assert Z.ndim == 3
    assert t.ndim == 2
    assert beta_t.ndim == 2
    t_batch, x_batch, n_atom, xyz = x.shape
    assert xyz == 3
    x = x.reshape(t_batch * x_batch, n_atom, 3)
    Z = Z.reshape(t_batch * x_batch, n_atom)
    t = t.reshape(t_batch * x_batch)
    beta_t = beta_t.reshape(t_batch * x_batch)
    assert x.shape[0] == t.shape[0] == beta_t.shape[0]

    t.requires_grad_()
    x.requires_grad_()

    if finite_diff:
        s, _ = model(x, Z, t)
        Δt = model.tau / model.N
        s_plus_Δs, _ = model(x, Z, t + Δt)
        ds_dt = (s_plus_Δs - s) / Δt
        assert ds_dt.shape == (t_batch * x_batch, n_atom, 3)
    else:
        with forward_ad.dual_level():
            t_plus_dt = forward_ad.make_dual(t, torch.ones_like(t))
            (s_plus_ds_dt, features) = model(x, Z, t_plus_dt)
            (s, ds_dt) = forward_ad.unpack_dual(s_plus_ds_dt)
        assert ds_dt.shape == (t_batch * x_batch, n_atom, 3)

    assert s.shape == x.shape
    ds_dx_trace = model.div(x, Z, t, use_hutchinson, n_hutchinson)
    if wo_drift:
        xs_plus_ss = torch.sum(s * s, dim=-1).sum(dim=-1)
    else:
        xs_plus_ss = torch.sum(s * (s + x), dim=-1).sum(dim=-1)
    for_nabla = xs_plus_ss + ds_dx_trace
    (nabla_for_nabla,) = torch.autograd.grad(
        for_nabla,
        x,
        grad_outputs=torch.ones_like(for_nabla),
        create_graph=True,
    )
    norm = torch.sqrt(
        torch.square(torch.einsum("ijk,i->ijk", nabla_for_nabla, beta_t / 2) - ds_dt)
        .mean(dim=-1)
        .mean(dim=-1)
    )
    return weight * loss_func(norm, torch.zeros_like(t))


def get_loss_data(
    model,
    t,
    x,
    Z,
    sigma_i,
    eps_i,
    weight: float = 1.0,
):
    if weight <= 0.0:
        return torch.tensor(0.0).to(x.device)
    x_batch, noise_batch, n_atom, xyz = x.shape
    x = x.reshape(noise_batch * x_batch, n_atom, 3)
    Z = Z.reshape(noise_batch * x_batch, n_atom)
    t = t.reshape(noise_batch * x_batch)
    eps_i = eps_i.reshape(noise_batch * x_batch, n_atom, 3)
    sigma_i = sigma_i.repeat(noise_batch, 1).squeeze(1)
    s, _ = model(x, Z, t)  # t_batch
    norm = s * sigma_i[:, None, None] + eps_i
    return norm.pow(2).sum(dim=-1).sum(dim=-1).mean() * weight
