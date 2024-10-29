"""
Transformer implementation for water molecule
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from moldiv.util import ResNetModule

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)


class Embed_T(nn.Module):
    def __init__(self, embed_t_dim, embed_dim):
        super().__init__()
        self.embed_t_dim = embed_t_dim
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_t_dim, embed_dim)

    def features_t(self, t: Tensor):
        n = self.embed_t_dim
        assert t.ndim == 1
        t = torch.unsqueeze(t, 1)
        emb_t0 = torch.cat(
            [
                t,
                torch.exp(-t * 3),
                torch.exp(-t * 12),
                torch.sin(torch.exp(-t * 6) * torch.tensor(math.pi)),
                torch.exp(-t * 6),
            ],
            dim=1,
        )
        if n <= 5:
            return emb_t0[:, 0:n]
        else:
            raise NotImplementedError

    def forward(self, t: Tensor):
        t = self.features_t(t)
        t = self.linear(t)
        return t


MAXPERIOD = 10000


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        max_period=MAXPERIOD,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] * 256  # 256 is my choice
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(torch.float32)
        return embeddings


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K, edge_types):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)  # (batch, natoms, natoms, 1)
        bias = self.bias(edge_types)  # (batch, natoms, natoms, 1)
        x = mul * x.unsqueeze(-1) + bias  # (batch, natoms, natoms, 1)
        x = x.expand(-1, -1, -1, self.K)  # (batch, natoms, natoms, kernel)
        mean = self.means.weight.float().view(-1)  # (kernel)
        std = self.stds.weight.float().view(-1).abs() + 1e-5  # (kernel)
        # (batch, natoms, natoms, kernel)
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.layer1(x))  # Activation Function
        x = self.layer2(x)
        return x


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, 1, bias=False
        )  # must be bias=False

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        """
        Args:
            query (Tensor): Invariant tensor with shape (n_batch, n_atom, embed_dim)
            attn_bias (Tensor): Invariant tensor with shape (n_batch, n_atom, n_atom,)
            delta_pos (Tensor): Equivariant tensor with shape (n_batch, n_atom, n_atom, 3)

        Returns:
            Tensor: Equivatiant score tensor with shape (n_batch, n_atom, 3)
        """
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query)
            .view(bsz, n_node, self.num_heads, self.head_dim)
            .transpose(1, 2)
            * self.scaling
        )  # [n_batch, num_head, n_atom, head_dim]
        k = (
            self.k_proj(query)
            .view(bsz, n_node, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(query)
            .view(bsz, n_node, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attn = q @ k.transpose(2, 3)  # [n_batch, num_head, n_atom, n_atom]
        attn_probs = softmax_dropout(
            attn.view(-1, n_node, n_node) + attn_bias, 0.1, self.training
        ).view(
            bsz, self.num_heads, n_node, n_node
        )  # [n_batch, num_head, n_atom, n_atom]
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [n_batch, num_head, n_atom, n_atom, 3]
        rot_attn_probs = rot_attn_probs.permute(
            0, 1, 4, 2, 3
        )  # [n_batch, num_head, 3, n_atom, n_atom]
        x = rot_attn_probs @ v.unsqueeze(2)  # [n_batch, num_head, 3, n_atom, head_dim]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        # [n_batch, n_atom, 3, num_head, head_dim] -> [n_batch, n_atom, 3, embed_dim]

        f1 = self.force_proj(x[:, :, 0, :]).view(bsz, n_node, 1)  # [n_batch, n_atom, 1]
        f2 = self.force_proj(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=2).float()  # [n_batch, n_atom, 3]
        return cur_force


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout=0.0,
        bias: bool = True,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query: Tensor, attn_bias: Tensor | None = None):
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        attn = self.out_proj(attn)
        return attn


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.self_attn = SelfMultiheadAttention(
            embed_dim, attention_heads, dropout=dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = ResNetModule(embed_dim, ffn_embed_dim, embed_dim, self.dropout)

    def forward(self, node_feature: Tensor, attn_bias: Tensor | None = None):
        x = node_feature
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.ffn(x)
        return x


class ScoreTransformer(nn.Module):
    """
    Time dependent Score base transformer

    Args:
        embed_dim (int): feature embed dimension
        ffn_embed_dim (int): Feed-Forward embed dimension
        n_layer (int): number of encoder layer
        attention_heads (int): Attention heads size
        num_kernerl (int): Number of embedding kernel
        N (int): number of discrete time steps
        dropout (float): The dropout
        n_block (int): Number of repeated blocks of tranformer encoder
        use_sin (bool): Use sinoidal positional embedding
        tau (float): max propagation time

    """

    def __init__(
        self,
        *,
        embed_dim: int = 64,
        ffn_embed_dim: int = 64,
        n_layer: int = 8,
        attention_heads: int = 16,
        num_kernel: int = 16,
        N: int = 500,
        dropout: float = 0.0,
        n_block: int = 1,
        use_sin: bool = True,
        tau: float = 1.0,
    ):
        super().__init__()
        self.blocks = n_block
        self.atom_types = 9  # H ~ O
        self.edge_types = self.atom_types**2
        self.input_dropout = 0.0
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.atom_encoder = nn.Embedding(self.atom_types, self.embed_dim, padding_idx=0)
        if use_sin:
            self.time_encoder = SinusoidalPositionEmbeddings(dim=embed_dim)
        else:
            self.time_encoder = Embed_T(embed_t_dim=5, embed_dim=embed_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim,
                    ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.final_ln = nn.LayerNorm(self.embed_dim)
        K = num_kernel
        self.gbf = GaussianLayer(K, self.edge_types)
        self.bias_proj = NonLinear(K, attention_heads)
        self.edge_proj = nn.Linear(K, embed_dim)
        self.node_proc = NodeTaskHead(embed_dim, attention_heads)
        self.N = N  # diffusion step
        self.tau = tau

    def forward(self, x: Tensor, Z: Tensor, t: Tensor):
        """
        Args:
            x (Tensor): absolute positions with shape (n_batch, n_atom, 3)
            Z (Tensor): atomic number with shape (n_batch, n_atom)
            t (Tensor): time with shape (n_batch,)

        Returns:
            tuple[Tensor, dict[int, Tensor]]: node(score), features

        """
        n_graph, n_node = Z.size()  # (batch, natoms)
        assert x.shape == (n_graph, n_node, 3), f"{x.shape=}"
        assert t.shape == (n_graph,)
        delta_pos = x.unsqueeze(1) - x.unsqueeze(2)
        assert delta_pos.shape == (n_graph, n_node, n_node, 3)
        dist: Tensor = torch.sqrt(
            torch.square(delta_pos).sum(dim=-1) + 1e-6
        )  # (batch, natoms, natoms, 3)
        assert dist.shape == (
            n_graph,
            n_node,
            n_node,
        )
        delta_pos = delta_pos / dist.unsqueeze(-1)

        edge_type = Z.view(n_graph, n_node, 1) * self.atom_types + Z.view(
            n_graph, 1, n_node
        )  # (batch, natoms, natoms)
        gbf_feature = self.gbf(dist, edge_type)

        enc_t = self.time_encoder(t).unsqueeze(1)  # (batch, 1, dim)
        graph_node_feature = (
            +self.atom_encoder(Z)  # (batch, atoms, dim)
            + self.edge_proj(
                gbf_feature.sum(dim=-2)
            )  # (batch, atoms, atoms, dim) -> (batch, atoms, dim)
            + enc_t
        )
        # logger.debug(f"{graph_node_feature.shape=}")
        output = F.dropout(
            graph_node_feature, p=self.input_dropout, training=self.training
        )
        output = output.transpose(0, 1).contiguous()  # (atoms, batch, dim)
        graph_attn_bias = (
            self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
        )  # (batch, heads, atoms, atoms)
        features: dict[int, Tensor] = {}
        features[0] = output
        # (batch*heads, atoms, atoms)
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for b in range(self.blocks):
            for i, enc_layer in enumerate(self.layers):
                output = enc_layer(output, attn_bias=graph_attn_bias)
                features[b * len(self.layers) + i + 1] = output
        output = self.final_ln(output)  # (atoms, batch, dim)
        output = output.transpose(0, 1)  # (batch, atoms, dim)
        node_output = self.node_proc(
            output, graph_attn_bias, delta_pos
        )  # (batch, atoms, 3)

        ###### Remove translational gradinent #############################
        node_output = node_output - node_output.mean(dim=1, keepdim=True)
        ###################################################################

        return node_output, features

    def div(
        self,
        x: Tensor,
        Z: Tensor,
        t: Tensor,
        use_hutchinson: bool = False,
        n_hutchinson: int = 20,
    ) -> Tensor:
        """
        Evaluate divergence ∇⋅s

        Args:
            x (Tensor): Positions
            Z (Tensor): Atomic Numbers
            t (Tensor): time

        """

        if use_hutchinson:
            raise NotImplementedError
        else:
            x.requires_grad_()
            s, _ = self(x, Z, t)
            div = torch.zeros_like(t)
            for i_atom in range(s.shape[1]):
                for j_xyz in range(s.shape[2]):
                    (div_ij,) = torch.autograd.grad(
                        s[:, i_atom, j_xyz],
                        x,
                        grad_outputs=torch.ones_like(t),
                        create_graph=True,
                    )
                    div += div_ij[:, i_atom, j_xyz]
        return div
