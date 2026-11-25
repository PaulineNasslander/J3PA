import math

import torch
from torch import nn
from torch.nn import functional as F

from constants import POINT_FEATURE_DIM, VOXEL_SIZE, XYZ_SLICE
from j3pa.glu import SwiGLUFFN
from j3pa.hilbert import encode_hilbert_bitwise
from j3pa.layers import VoxelSizeEmbedding, log_voxel_size_coordinate
from j3pa.positional_encodings import PEDynamic
from j3pa.qk_layernorm import QK_LAYERNORM_IMPLS
from j3pa.rope import RoPE
from j3pa.zorder import encode_zorder_fast_3d

from torch.utils.checkpoint import checkpoint

# Size of the positional encoding of the input positions.
INPUT_PE_DIM = 64

class DynTanhNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._alpha = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch.tanh(self._alpha * x)

def get_grid_idx(
    pt_pos: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_size: float | torch.Tensor,
    batch_size: int | None = None,
    min_pos: torch.Tensor | None = None,
):
    """Get the integer grid index of each point.

    Args:
        pt_pos: (N, 3) point positions.
        batch_idx: (N,) batch indices.
        cell_size: float, (N,) per-point or (N, 3) per-point and per-axis cell sizes.
        batch_size: Optional number of samples in the batch.
        min_pos: Optional (B, 3) per-sample grid origins. The per-sample minimum positions if not given.

    Returns:
        grid_idx: (N, 3) and min_pos: (B, 3) the per-sample grid origins.
    """
    B = batch_size if batch_size is not None else batch_idx.max().item() + 1
    if min_pos is None:
        min_pos: torch.Tensor = torch.empty((B, 3), device=pt_pos.device, dtype=pt_pos.dtype)
        #torch.scatter_reduce_ seems to not broadcast therefore we must exand batch_idx to 3 channels manually here.
        min_pos.scatter_reduce_(0, batch_idx.unsqueeze(-1).expand(-1,3), pt_pos, reduce="amin", include_self=False)
    else:
        assert min_pos.device == pt_pos.device
        assert min_pos.shape[0] == B
        assert min_pos.shape[-1] == pt_pos.shape[-1]
    if isinstance(cell_size, torch.Tensor) and cell_size.dim() == 1:
        cell_size = cell_size.unsqueeze(-1)
    return torch.div(pt_pos - min_pos[batch_idx], cell_size, rounding_mode="trunc").long(), min_pos


def space_filling_idx(grid_idx):
    # Technically it could be possible to use a different bit depth for every entry in the batch.
    # PTV3 calculates on global bit depth, and it's easier to work this way.
    num_bits = grid_idx.max().item().bit_length()
    # (N)
    hilbert = encode_hilbert_bitwise(grid_idx, num_bits=num_bits)
    zorder = encode_zorder_fast_3d(grid_idx, num_bits=num_bits)
    trans_hilbert = encode_hilbert_bitwise(grid_idx[..., [1, 0, 2]], num_bits=num_bits)
    trans_zorder = encode_zorder_fast_3d(grid_idx[..., [1, 0, 2]], num_bits=num_bits)

    # (4, N)
    return torch.stack([hilbert, zorder, trans_hilbert, trans_zorder], dim=0)


def batched_permute(sf_idx, batch_idx, batch_size: int | None = None):
    assert sf_idx.shape[-1] == batch_idx.shape[-1]
    B = batch_size if batch_size is not None else batch_idx.max().item() + 1

    # Sad that randperm is not vectorized :--(
    # (B, 4) This shape so we can index it easily below.
    permutations = torch.argsort(torch.rand(B, 4, device=sf_idx.device), dim=1)

    # (B, 4) -> (N, 4)
    batched_permutations = permutations[batch_idx]

    # (4, N)
    return torch.gather(sf_idx, dim=0, index=batched_permutations.T)


def _scatter_softmax_reduce_inplace(x: torch.Tensor, dim, idx, src, softmax_src, include_self=False) -> torch.Tensor:
    """See torch scatter_reduce documentation.

    Assumes all locations are scattered to, so empty locations will be used.
    
    Args:
        x: (M, C)
        idx: (N,) in range [0, M)
        src: (N, C)
        softmax_src: (N, C)
    """
    # (M, C)
    m = torch.empty_like(x)
    m.scatter_reduce_(dim, idx, softmax_src, reduce="amax", include_self=False)

    # (M, C)
    m_expanded = torch.gather(m, dim=dim, index=idx).float()
    src_exp = (softmax_src.float() - m_expanded).exp()
    s = torch.empty_like(x, dtype=torch.float)
    s.scatter_reduce_(dim, idx, src_exp, reduce="sum", include_self=False)
    s_expanded = torch.gather(s, dim=dim, index=idx)
    softmax = (src_exp/s_expanded).to(x.dtype)
    x.scatter_reduce_(dim, idx, softmax*src, include_self=include_self, reduce="sum")

class PointSelfAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion_factor: int | float,
        group_size: int,
        zero_residual=False,
        weight_mix_qk: float = 0.0,
        glu: bool = False,
        register_reduction: str = "amax",
        gradient_checkpointing: bool = False,
        sep_reg_voxel_norms: bool = False,
        use_attention_bias: bool = False,
        rope_layer: bool = None,
        qk_layernorm: bool = False,
        qkv_bias: bool = False,
        condition_voxel_size: bool = False,
        xsa: bool = False,
        sdpa_gate: bool = False,
        qk_layernorm_impl: str = "torch",
    ):
        assert dim % num_heads == 0
        assert 0 <= weight_mix_qk <= 1.0
        assert register_reduction in ("amax", "sum", "mean", "softmax"), f"Unknown register reduction method {register_reduction}."
        self._register_reduction = register_reduction

        super().__init__()
        self._ln_attn = nn.LayerNorm(dim)
        self._sep_reg_voxel_norms = sep_reg_voxel_norms
        if sep_reg_voxel_norms:
            self._ln_attn_reg = nn.LayerNorm(dim)
            self._ln_ffn_reg = nn.LayerNorm(dim)

        if condition_voxel_size:
            # Per-layer offset on the stage's shared voxel size modulation (adaLN-single). Zero is unmodulated.
            # Each layer has its own bias for the voxel size modulation. This allows the model to shift its modulation depending 
            # on layer while keeping the number of params down by sharing the main voxel size modulation across layers.
            self._adaln_vs_bias = nn.Parameter(torch.zeros(6, dim))
        else:
            self._adaln_vs_bias = None

        self._pqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self._q_ln = nn.LayerNorm(dim//num_heads) if qk_layernorm else None
        self._k_ln = nn.LayerNorm(dim//num_heads) if qk_layernorm else None
        # Only selects the function computing the QK-LayerNorm with the parameters of _q_ln and _k_ln. It holds no
        # state, so checkpoints load with any implementation (see j3pa/qk_layernorm.py).
        assert qk_layernorm_impl in QK_LAYERNORM_IMPLS, f"Unknown QK-LayerNorm implementation {qk_layernorm_impl}."
        self._qk_layernorm_impl = qk_layernorm_impl
        # Exclusive self attention (XSA): remove the component of the attention output along the token's own value.
        self._xsa = xsa
        self._rope = rope_layer
        self._use_attention_bias = use_attention_bias
        if use_attention_bias:
            # (1, H, 1, C//H) 
            self.k_bias = nn.Parameter(torch.zeros(1, num_heads, 1, dim // num_heads), requires_grad=True)
            self.v_bias = nn.Parameter(torch.zeros(1, num_heads, 1, dim // num_heads), requires_grad=True)

            #TODO could do better here, i just copied these init values from the paper. 
            torch.nn.init.normal_(self.k_bias, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.v_bias, mean=0.0, std=0.02)
            
        self._po = nn.Linear(dim, dim)
        # Elementwise sigmoid gate on the attention output before the output projection (SDPA output gating).
        self._sdpa_gate_lin = nn.Linear(dim, dim) if sdpa_gate else None
        self._num_heads = num_heads
        self._group_size = group_size

        self._ln_ffn = nn.LayerNorm(dim)
        if glu:
            self._ffn = SwiGLUFFN(dim=dim, ff_mult=expansion_factor, nearest_multiplier=64)
        else:
            self._ffn = nn.Sequential(
                nn.Linear(dim, expansion_factor * dim, bias=(expansion_factor > 1)),
                nn.GELU(),
                nn.Linear(expansion_factor * dim, dim),
            )

        # Symmetry in attentions nonlinearity (softmax) makes Glorot initialization preferrable to
        # make QKV projections. We can keep Kaiming in the ffn.
        nn.init.xavier_uniform_(self._pqkv.weight)
        if qkv_bias:
            nn.init.zeros_(self._pqkv.bias)

        if zero_residual:
            # Start with clean residuals like modern LLM trainings.
            nn.init.zeros_(self._po.weight)
            nn.init.zeros_(self._po.bias)
            if not glu:
                nn.init.zeros_(self._ffn[-1].weight)
                nn.init.zeros_(self._ffn[-1].bias)
            else:
                nn.init.zeros_(self._ffn._lin3.weight)
                nn.init.zeros_(self._ffn._lin3.bias)
        else:
            nn.init.xavier_uniform_(self._po.weight)
            nn.init.zeros_(self._po.bias)

        if weight_mix_qk > 0:
            self._pqkv.weight.data[dim : 2 * dim] = (
                weight_mix_qk * self._pqkv.weight.data[:dim] + (1.0 - weight_mix_qk) * self._pqkv.weight.data[dim : 2 * dim]
            )
        self._gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        pt_feat,
        order_idx,
        inv_order_idx,
        pad_idx,
        unpad_idx,
        pt_pos = None,
        register = None,
        batch_indices = None,
        voxel_size_mod = None,
        batch_size = None,
    ):
        """
        Args:
            pt_feat: (N, C)
            order_idx: (N)
            inv_order_idx: (N)
            pad_idx: (M) M>=N
            unpad_idx: (N)
            pt_pos: (N, 3)
            register: (B, R, C)
            batch_indices: (N,). This is only needed when we use register tokens or voxel size conditioning
                           so that we can divide their update properly across the batch. In the registerless
                           case batching is implicitly handled by the pad and unpad indices.
                           Registers and voxel size conditioning can not be used at the same time.
            voxel_size_mod: (B, 6, C) voxel size modulation shared by the stage, the deltas of
                            (alpha_attn, beta_attn, gamma_attn, alpha_ffn, beta_ffn, gamma_ffn) from identity.
            batch_size: batch_size.
        """
        assert pad_idx.shape[0] % self._group_size == 0
        assert pt_feat.shape[0] == order_idx.shape[0]
        assert pt_feat.shape[0] == inv_order_idx.shape[0]
        assert pt_feat.shape[0] == unpad_idx.shape[0]
        assert pt_pos.shape[-1] == 3
        assert len(pt_pos.shape) == len(pt_feat.shape)
        if self._rope is not None:
            assert pt_pos is not None
        else:
            assert pt_pos is None
        if batch_indices is not None:
            assert batch_indices.shape[0] == pt_feat.shape[0]
        if register is not None:
            if self._adaln_vs_bias is not None:
                raise NotImplementedError("registers and voxelsize conditioning are not supported at the same time for now")
            assert len(register.shape) == 3
            assert register.shape[-1] == pt_feat.shape[-1]
            assert batch_indices is not None

        N, C = pt_feat.shape
        use_registers = register is not None
        R = register.shape[1] if use_registers else None

        if self._adaln_vs_bias is not None:
            assert voxel_size_mod is not None
            assert batch_indices is not None
            mods = voxel_size_mod + self._adaln_vs_bias
            if torch.is_autocast_enabled("cuda"):
                mods = mods.to(torch.get_autocast_dtype("cuda"))
            # Deltas from identity, applied as x + delta * x: 1 + delta would round small deltas away in bfloat16.
            #alpha_attn, beta_attn, gamma_attn, alpha_ffn, beta_ffn, gamma_ffn = mods[batch_indices].unbind(1)
            B = mods.shape[0]
            # (N, B)
            oh = F.one_hot(batch_indices, B).to(mods.dtype)
            # (N, B) @ (B, 1) -> (N, 1)
            mods_pt = (oh @ mods.reshape(B, -1)).view(batch_indices.shape[0], *mods.shape[1:])
            alpha_attn, beta_attn, gamma_attn, alpha_ffn, beta_ffn, gamma_ffn = mods_pt.unbind(1)
        else:
            alpha_attn, beta_attn, gamma_attn, alpha_ffn, beta_ffn, gamma_ffn = (None,) * 6

        if self._gradient_checkpointing:
            combined_feat = checkpoint(
                self._attn_forward,
                pt_feat=pt_feat,
                order_idx=order_idx,
                inv_order_idx=inv_order_idx,
                pad_idx=pad_idx,
                unpad_idx=unpad_idx,
                pt_pos=pt_pos,
                register=register,
                batch_indices=batch_indices,
                alpha_attn=alpha_attn,
                beta_attn=beta_attn,
                gamma_attn=gamma_attn,
                use_reentrant=False,
                preserve_rng_state=False
            )
            return checkpoint(
                self._ffn_forward, combined_feat, N, R, use_registers, alpha_ffn=alpha_ffn, beta_ffn=beta_ffn, gamma_ffn=gamma_ffn,
                use_reentrant=False, preserve_rng_state=False,
            )

        combined_feat = self._attn_forward(
            pt_feat=pt_feat,
            order_idx=order_idx,
            inv_order_idx=inv_order_idx,
            pad_idx=pad_idx,
            unpad_idx=unpad_idx,
            pt_pos=pt_pos,
            register=register,
            batch_indices=batch_indices,
            alpha_attn=alpha_attn,
            beta_attn=beta_attn,
            gamma_attn=gamma_attn,
        )
        return self._ffn_forward(combined_feat, N, R, use_registers, alpha_ffn=alpha_ffn, beta_ffn=beta_ffn, gamma_ffn=gamma_ffn)

    def _qk_layernorm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LayerNorm of q and k over the head dim, returned in their dtype."""
        return QK_LAYERNORM_IMPLS[self._qk_layernorm_impl](q, k, self._q_ln, self._k_ln)

    def _attn_forward(
        self, pt_feat, order_idx, inv_order_idx, pad_idx, unpad_idx, pt_pos, register=None, batch_indices=None,
        alpha_attn=None, beta_attn=None, gamma_attn=None,
    ):
        num_groups = pad_idx.shape[0] // self._group_size
        N, C = pt_feat.shape

        normed_feat = self._ln_attn(pt_feat)
        if gamma_attn is not None:
            assert beta_attn is not None
            # Deltas from identity, applied as x + delta * x. having as before with x*(1 + delta) could cause rounding errors in bfloat16.
            normed_feat = normed_feat + gamma_attn * normed_feat + beta_attn

        fused_pad_idx = order_idx[pad_idx]
        # Gate scores are query-side, so they must come from the same normalized input as q.
        pt_gate = torch.sigmoid(self._sdpa_gate_lin(normed_feat)) if self._sdpa_gate_lin is not None else None

        if self._rope:
            qkv = self._pqkv(normed_feat)[fused_pad_idx]
            q, k, v = (
                qkv.view(num_groups, self._group_size, 3, self._num_heads, C // self._num_heads)
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )

            if self._q_ln is not None and self._qk_layernorm_impl == "triton_fused_rope":
                from j3pa.qk_layernorm_rope_triton import qk_layernorm_rope

                assert self._k_ln is not None
                q, k = qk_layernorm_rope(q, k, self._q_ln, self._k_ln, self._rope, pt_pos[fused_pad_idx])
            else:
                # QK-LayerNorm must precede RoPE.
                if self._q_ln is not None:
                    assert self._k_ln is not None
                    q, k = self._qk_layernorm(q, k)

                cos, sin = self._rope.cos_sin(pt_pos[fused_pad_idx], qkv.dtype)
                head_shape = (num_groups, self._group_size, self._num_heads, C // self._num_heads)
                cos = cos.view(head_shape).permute(0, 2, 1, 3)
                sin = sin.view(head_shape).permute(0, 2, 1, 3)
                q = self._rope.apply(q, cos, sin)
                k = self._rope.apply(k, cos, sin)
                del cos, sin
            del qkv

        else:
            qkv = self._pqkv(normed_feat)[fused_pad_idx]
            # K : number of groups
            # G: group size
            # (K*G, 3C) -> (K, G, 3, H, C//H) -> 3 x (K, H, G, C//H)
            q, k, v = (
                qkv.view(
                    num_groups,
                    self._group_size,
                    3,
                    self._num_heads,
                    C // self._num_heads,
                )
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )
            if self._q_ln is not None:
                assert self._k_ln is not None
                q, k = self._qk_layernorm(q, k)
            del qkv

        fused_unpad_idx = unpad_idx[inv_order_idx]

        if self._use_attention_bias:
            # Point groups are of shape (K, H, G, C//H) and the biases are head specific (1, H, 1, C//H) 
            # and should participate in each of the K groups. 
            # All groups attend to the same bias.
            k_bias = self.k_bias.repeat(num_groups, 1, 1, 1) # (K, num_heads, 1, dim // num_heads)
            v_bias = self.v_bias.repeat(num_groups, 1, 1, 1)

            # Should work to just concat these here and not change anything for the 
            # register code as the queries are untouched.
            k = torch.cat((k_bias, k), dim=2)
            v = torch.cat((v_bias, v), dim=2)

        # Registers currently don't work with batch indices!!!!
        if register is not None:
            B, R, C = register.shape

            # Why is this somehow messed up if we don't do this the pooling scatter complains.
            register = register.to(dtype=pt_feat.dtype)

            # (B, R, 3C) -> (B, R, 3, H, C//H) -> 3, (B, H, R, C//H)
            ln_attn_reg = self._ln_attn_reg if self._sep_reg_voxel_norms else self._ln_attn
            reg_normed = ln_attn_reg(register)
            rq, rk, rv = self._pqkv(reg_normed).view(B, R, 3, self._num_heads, C // self._num_heads).permute(2, 0, 3, 1, 4).unbind(0)

            if self._q_ln is not None:
                rq, rk = self._qk_layernorm(rq, rk)

            # Point groups are of shape (K, H, G, C//H) and the registers should participate in
            # each of the K groups. We need to make sure that each register only participates in
            # its own sample and not the others in the batch.
            # (B,)
            batch_counts = torch.empty(B, dtype=batch_indices.dtype, device=batch_indices.device)
            batch_counts.scatter_reduce_(0, batch_indices, torch.ones_like(batch_indices), reduce="sum", include_self=False)
            num_groups_per_sample = (batch_counts + self._group_size - 1) // self._group_size
            # padded_counts = num_groups_per_sample*self._group_size
            batch_to_group = torch.arange(B, dtype=batch_indices.dtype, device=batch_indices.device).repeat_interleave(num_groups_per_sample)

            # (B -> K): (B, H, R, C//H) -> (N//K, H, R, C//H)
            rq = rq[batch_to_group]
            rk = rk[batch_to_group]
            rv = rv[batch_to_group]

            # TODO concatenating along the -2 axis forces noncontiguous memory. Can we do the
            # ordering different and get the same result?
            q = torch.cat([q, rq], dim=-2)
            k = torch.cat([k, rk], dim=-2)
            v = torch.cat([v, rv], dim=-2)

            # (N//K, H, K + R, C//H)
            attn_out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            if self._xsa:
                attn_out = self._exclude_self(attn_out, v)
            del q, k, v
            # (N//K, H, K + R, C//H) -> (N//K, K + R, H, C//H) -> (N//K, K + R, C)
            attn_out = attn_out.transpose(1, 2).contiguous().view(num_groups, self._group_size + R, C)
            # (N//K, K + R, C) -> (N//K, K, C), (N//K, R, C)
            # (N//K, K, C) -> (N//K*K, C) -> (N, C)
            attn_out_pt = attn_out[:, : self._group_size, :].contiguous().view(-1, C)[fused_unpad_idx]
            if pt_gate is not None:
                attn_out_pt = attn_out_pt * pt_gate

            # (N//K, R, C) -> (B, R, C)
            # We need to somehow combine the updates from all the groups into one by reducing over
            # the first axis.
            # Now we must scatter reduce it with the correct batch indices
            # (N//K, R, C)
            attn_out_r = attn_out[:, self._group_size :, :]

            register_attn_update = torch.empty_like(register)  # (B,
            # (N//K) -> (N//K, R, C)
            scatter_indices = batch_to_group.view(-1, 1, 1).expand(-1, R, C)
            # (N//K, R, C) -> (B, R, C)

            if self._register_reduction == "softmax":
                _scatter_softmax_reduce_inplace(register_attn_update, 0, scatter_indices, attn_out_r, attn_out_r, include_self=False)
            else:
                register_attn_update.scatter_reduce_(0, scatter_indices, attn_out_r, reduce=self._register_reduction, include_self=False)
            del attn_out_r
            if self._sdpa_gate_lin is not None:
                register_attn_update = register_attn_update * torch.sigmoid(self._sdpa_gate_lin(reg_normed))

            # Concatenate the reduced register and output features again so we can project
            # everything in one go.
            feat = torch.cat([pt_feat, register.view(-1, C)], dim=0)
            attn_out = torch.cat([attn_out_pt, register_attn_update.view(-1, C)], dim=0)

            return feat + self._po(attn_out)
        
        # No registers
        # (K, H, G, C//H)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        if self._xsa:
            attn_out = self._exclude_self(attn_out, v)
        del q, k, v
        # (K, H, G, C//H) -> (K*G, C)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(num_groups * self._group_size, C)[fused_unpad_idx]
        if pt_gate is not None:
            attn_out = attn_out * pt_gate
        attn_out = self._po(attn_out)
        if alpha_attn is not None:
            return pt_feat + attn_out + alpha_attn * attn_out
        return pt_feat + attn_out

    def _exclude_self(self, attn_out, v):
        """XSA: remove the component of every query's attention output along its own value vector, per head.

        Kept in fp32: the subtraction cancels almost all of attn_out, so bf16 intermediates destroy most of the
        orthogonality this is supposed to establish.
        """
        # With an attention bias the first key and value are the bias, which has no query.
        v = v[..., 1:, :] if self._use_attention_bias else v
        assert v.shape == attn_out.shape
        v_normed = F.normalize(v.float(), p=2.0, dim=-1)
        out = attn_out.float()
        return (out - (out * v_normed).sum(dim=-1, keepdim=True) * v_normed).to(attn_out.dtype)
    

    def _ffn_forward(self, combined_feat, N, R, use_register: bool, alpha_ffn=None, beta_ffn=None, gamma_ffn=None):
        """
        
            N: Number of voxel features
            R: Number of registers.
            alpha_ffn, beta_ffn, gamma_ffn: (N, C) voxel size modulation deltas from identity.
        """
        if use_register and self._sep_reg_voxel_norms:
            normed = torch.cat([self._ln_ffn(combined_feat[:N]), self._ln_ffn_reg(combined_feat[N:])], dim=0)
        else:
            normed = self._ln_ffn(combined_feat)

        if alpha_ffn is None:
            combined_feat = combined_feat + self._ffn(normed)
        else:
            assert beta_ffn is not None and gamma_ffn is not None
            ffn_out = self._ffn(normed + gamma_ffn * normed + beta_ffn)
            combined_feat = combined_feat + ffn_out + alpha_ffn * ffn_out
        if use_register:
            C = combined_feat.shape[-1]
            return combined_feat[:N], combined_feat[N:].view(-1, R, C)
        return combined_feat, None
    

def pad_and_unpad_idx(batch_indices: torch.Tensor, group_size: int, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    B = batch_size if batch_size is not None else batch_indices.max().item() + 1
    counts = torch.empty((B,), device=batch_indices.device, dtype=batch_indices.dtype)
    counts.scatter_reduce_(0, batch_indices, torch.ones_like(batch_indices), reduce="sum", include_self=False)
    padded_counts = ((counts + group_size - 1) // group_size) * group_size

    pad_idx = torch.arange(padded_counts.sum(), device=batch_indices.device, dtype=batch_indices.dtype)

    # TODO: Does this work in our predictor as well?
    pad_batch_indices = torch.arange(B, device=batch_indices.device, dtype=batch_indices.dtype).repeat_interleave(padded_counts)
    pad_batch_offsets = (torch.cumsum(counts, 0) - counts)[pad_batch_indices]
    pad_offsets = (torch.cumsum(padded_counts, 0) - padded_counts)[pad_batch_indices]
    pad_idx -= pad_offsets

    counts_per_point = counts[pad_batch_indices]
    unpad_mask = pad_idx < counts_per_point
    # last_group_start = torch.where(counts_per_point > group_size, (counts_per_point//group_size - 1) * group_size, 0)
    # last_group_start = (counts_per_point // group_size)* group_size
    last_group_start = torch.clamp(counts_per_point - group_size, min=0)

    pad_idx = torch.where(unpad_mask, pad_idx, last_group_start + (pad_idx % counts_per_point))

    # Now we need to add the offset (start index) for each batch. These are based on the original
    # counts because we are indexing the original point cloud tensor. It needs to be the size of
    # the padded tensor however
    unpad_idx = pad_idx + pad_offsets
    unpad_idx = unpad_idx[unpad_mask]
    pad_idx += pad_batch_offsets
    # Now we need to make the unpadding indices
    # These are just
    # We need the pad_batch_offsets to index every batch, and then index Ni elements after that
    # for every sample
    return pad_idx, unpad_idx


class Pool(nn.Module):
    def __init__(self, stride: int, in_dim: int, out_dim: int, activation=nn.GELU(), postnorm: bool = False):  # noqa: B008
        super().__init__()
        self._n_bits_to_shift = math.log2(stride)
        assert self._n_bits_to_shift.is_integer()
        self._n_bits_to_shift = int(self._n_bits_to_shift)
        self._proj = nn.Linear(in_dim, out_dim)
        self._norm = nn.LayerNorm(out_dim) if postnorm else nn.LayerNorm(in_dim)
        self._out_dim = out_dim
        self._activation = activation
        self._stride = stride
        self._postnorm = postnorm

    def forward(self, pt_feat, pt_pos, sf_idx, batch_idx, batch_size: int | None = None):
        """

        This also permutes the space-filling curves randomly so that next block can use them
        directly.

        Returns:
            new_feat, avg_pos, pooled_batch_idx, new_sf_idx, new_order, new_inv_order, up_index
        """
        # Shift by 1 bit for each dimension (3) to increase voxel size by factor 2 along each
        # dimension.
        pooled_sf_idx = sf_idx >> self._n_bits_to_shift * 3

        # Make linear indices together with the batch indices.
        # We compute all (4, N) of them instead of just N since we will use the linear index to do
        # the sorting later on to guarantee batch independence in the sort.
        # Important to offset by 1 when making linear indices.
        max_sf_idx = pooled_sf_idx.max() + 1
        linear_idx = max_sf_idx * batch_idx + pooled_sf_idx

        # pooled_sf_idx (4, N)
        # unique_pooled_idx (K)  K <= N
        # up_idx (N)
        unique_pooled_idx, up_idx = torch.unique(linear_idx[0], sorted=False, return_inverse=True, return_counts=False)

        if self._postnorm:
            pt_feat = self._proj(pt_feat)
        else:
            pt_feat = self._proj(self._norm(pt_feat))  # for init pooling this projection takes the pt_feats to bfloat16 must be placed here.
        out_feat = torch.zeros(
            unique_pooled_idx.shape[0],
            self._out_dim,
            device=pt_feat.device,
            dtype=pt_feat.dtype,
        )

        # (N, C) -> (K, D)
        out_feat.scatter_reduce_(
            0,
            up_idx.unsqueeze(-1).expand(-1, out_feat.shape[-1]),
            pt_feat,
            reduce="amax",
            include_self=False,
        )

        # ??? Does this activation suck???
        if self._postnorm:
            # Layernorm outputs fp32 and we want a bfloat16 residual stream.
            out_feat = self._norm(out_feat).to(dtype=out_feat.dtype)
        else:
            out_feat = self._activation(out_feat)

        # (N, 3) -> (K, 3)
        with torch.no_grad():
            out_pos = torch.empty(
                unique_pooled_idx.shape[0],
                pt_pos.shape[-1],
                device=pt_pos.device,
                dtype=pt_pos.dtype,
            )
            out_pos.scatter_reduce_(
                0,
                up_idx.unsqueeze(-1).expand(-1, out_pos.shape[-1]),
                pt_pos,
                reduce="mean",
                include_self=False,
            )

            # (N,) -> (K)
            out_batch = torch.empty(
                unique_pooled_idx.shape[0],
                device=batch_idx.device,
                dtype=batch_idx.dtype,
            )
            out_batch.scatter_reduce_(0, up_idx, batch_idx, reduce="amax", include_self=False)

        # Several points get mapped to a unique index. We want to have the corresponding indices of
        # each space-filling curve for the unique clusters. We get these by finding the first
        # occurence of each original index in the up indices. If we index each first occurence in
        # the pooled space-filling curves we will get the space filling indices for the next
        # transformer stage.
        # We can safely scatter into an empty array because each unique index will get hit by up_idx
        # at least once.
        first_occurrence = torch.empty(unique_pooled_idx.shape[0], device=up_idx.device, dtype=up_idx.dtype)
        first_occurrence.scatter_reduce_(0, up_idx, torch.arange(up_idx.shape[0], device=up_idx.device), reduce="amin", include_self=False)
        pooled_sf_idx = pooled_sf_idx[..., first_occurrence]

        # Sort and permute the pooled space-filling curves.
        # Note that in PTV3 this is not properly batched and each entry in the batch gets the same
        # permutation. We properly batch it with different permutations for each sample.
        # Only permute during training and not during inference so we get nondeterminism.
        if self.training:
            pooled_sf_idx = batched_permute(pooled_sf_idx, out_batch, batch_size=batch_size)
        pooled_linear_idx = max_sf_idx * out_batch + pooled_sf_idx

        # (4, N)
        # This order needs to keep the batches separate. Use linear_idx, which will guarantee this
        # and still maintain the correct space-filling order on each batch.
        order = torch.argsort(pooled_linear_idx, dim=-1)
        inv_order = torch.empty_like(order)
        inv_order.scatter_(
            -1,
            order,
            torch.arange(order.size(-1), device=order.device).expand_as(order),
        )
        

        return out_feat, out_pos, out_batch, pooled_sf_idx, order, inv_order, up_idx


class UnPool(nn.Module):
    def __init__(self, dim_lowres, dim_highres, activation=nn.GELU(), lowres_droprate = 0.0):  # noqa: B008
        super().__init__()

        # Affine norm can absorb the bias
        self._proj = nn.Linear(dim_lowres, dim_highres, bias=False)

        # PTV has batchnorm here.
        self._norm = nn.LayerNorm(dim_lowres)

    def forward(self, feat_lowres, feat_highres, up_idx):
        # In PTV they would norm after proj
        return feat_highres + self._proj(self._norm(feat_lowres))[up_idx]


def legacy_layer_split(num_layers_per_stage: list[int], out_norm_stages: list[int], downsample_only: bool = False) -> tuple[list[int], list[int]]:
    """Down- and up-path layer counts of the former total number of layers per stage.

    The total included one layer after the up connection for every stage but the deepest. Only the stages in
    out_norm_stages have an up connection, so the other stages lost that layer.
    """
    if downsample_only:
        return list(num_layers_per_stage), [0] * (len(num_layers_per_stage) - 1)
    num_down = [n - 1 for n in num_layers_per_stage[:-1]] + [num_layers_per_stage[-1]]
    num_up = [int(s in out_norm_stages) for s in range(len(num_layers_per_stage) - 1)]
    return num_down, num_up


class PointTransformer(nn.Module):
    def __init__(
        self,
        stage_dims: list[int] = [64, 128, 256, 512],
        num_layers_per_stage: list[int] | None = None,
        num_heads_per_stage: list[int] = [2, 4, 8, 16],
        group_size: int = 1024,
        downsample_only: bool = False,
        voxel_size: float = VOXEL_SIZE,
        num_registers: int = 0,
        register_reduction: str = "amax",
        out_norm_stages: list = [],  # noqa: B006
        sep_reg_voxel_norms: bool = False,
        use_attention_bias: bool = False,
        qk_layernorm: bool = True,
        return_down_path: bool = False,
        condition_voxel_size: bool = False,
        voxel_size_point_feature: bool = False,
        pos_norm_voxel_size: float | None = None,
        voxel_size_range: tuple[float, float] | None = None,
        voxel_size_cond_init_std: float = 0.0,
        deepest_mid_target: bool = False,
        num_down_layers_per_stage: list[int] | None = None,
        num_up_layers_per_stage: list[int] | None = None,
        xsa: bool = False,
        sdpa_gate: bool = False,
        pool_postnorm: bool = False,
        qkv_bias: bool = False,
        qk_layernorm_impl: str = "torch",
    ):
        """Point Transformer backbone

        Args:
            stage_dims: channel dimensions of each stage.
            num_layers_per_stage: Former way to set the depth, as an alternative to num_down_layers_per_stage and
                                  num_up_layers_per_stage: the total number of layers in each stage, of which the
                                  last one comes after the up connection (see legacy_layer_split). Defaults to
                                  [4, 4, 4, 4] if no depth is given.
            group_size (int, optional): Size of attention groups. Defaults to 1024.
            downsample_only (bool, optional): Whether to use only the encoder part.
                This only uses the downsampling layers no upsampling layers.
                Defaults to False.
            voxel_size: Default voxel size, used whenever no voxel size is passed to forward.
            num_registers: Number of global register tokens.
            out_norm_stages: Stage indices for which we will apply a layernorm at the output. These
                             must have a loss function as we will get problems with DDP otherwise.
            return_down_path: Build layernorms for the down-path (pre-decoder) features of the decoded
                              out_norm_stages so that forward can return them. These also need a loss.
            condition_voxel_size: Modulate all attention layers with the voxel size, through one shared modulation
                                  per stage plus a learned per-layer offset (adaLN-single). Starts unmodulated.
            voxel_size_point_feature: Append the normalised log voxel size as an extra feature to every input point,
                                      after the raw point features (time if used, xyz in meters and raw intensity)
                                      and the positional encoding of the position in voxels.
            pos_norm_voxel_size: Fixed voxel size to normalise positions by (input PE and RoPE). None uses
                                 each sample's voxel size. The grid always uses each sample's voxel size.
            voxel_size_range: (min, max) training voxel sizes, mapped to u in [-1, 1] for the voxel size inputs. None
                              (no voxel size sampling) gives u = log2(voxel_size / default voxel size), i.e. 0 at the
                              default and +-1 per factor 2. Must match between training and evaluation.
            voxel_size_cond_init_std: Init std of the layer producing the shared voxel size modulations.
            deepest_mid_target: With return_down_path, also return the deepest stage's features halfway
                                through its layers (normalized) as its down-path entry. These also need a loss.
            num_down_layers_per_stage: Number of layers of each stage before its up connection, i.e. on the down
                                       path. All layers of the deepest stage are on the down path.
            num_up_layers_per_stage: Number of layers of each stage but the deepest after its up connection, i.e.
                                     on the up path. Must be 0 for stages without an up connection, which are the
                                     ones not in out_norm_stages and all of them with downsample_only.
            xsa: Exclusive self attention in every attention layer: the component of a token's attention output
                 along its own value vector is removed.
            sdpa_gate: Gate the attention output of every attention layer elementwise with a sigmoid of a linear
                       map of the token's normalized input, before the output projection.
            qk_layernorm_impl: Implementation computing the QK-LayerNorm, a key of QK_LAYERNORM_IMPLS in
                               j3pa/qk_layernorm.py. The parameters are the same for every implementation, so
                               checkpoints load with any of them. "triton_fused_rope" also fuses the point tokens'
                               RoPE; registers and layers without RoPE use the norm-only Triton implementation.
        """
        if num_down_layers_per_stage is None and num_up_layers_per_stage is None:
            num_layers_per_stage = [4, 4, 4, 4] if num_layers_per_stage is None else num_layers_per_stage
            assert len(num_layers_per_stage) == len(stage_dims)
            if downsample_only:
                assert all(nl > 1 for nl in num_layers_per_stage)
            num_down_layers_per_stage, num_up_layers_per_stage = legacy_layer_split(num_layers_per_stage, out_norm_stages, downsample_only)
        else:
            assert num_layers_per_stage is None, "Give either the total number of layers per stage or the down and up layers per stage."
        assert num_down_layers_per_stage is not None and num_up_layers_per_stage is not None, "Give both the down and the up layers per stage."
        assert len(num_down_layers_per_stage) == len(stage_dims), "Need the number of down layers of every stage."
        assert len(num_up_layers_per_stage) == len(stage_dims) - 1, "Need the number of up layers of every stage but the deepest."
        assert all(n >= 0 for n in num_down_layers_per_stage) and all(n >= 0 for n in num_up_layers_per_stage)
        assert all(n == 0 for s, n in enumerate(num_up_layers_per_stage) if downsample_only or s not in out_norm_stages), "Only stages with an up connection can have up layers."
        assert all(d + u > 0 for d, u in zip(num_down_layers_per_stage, [*num_up_layers_per_stage, 0])), "Every stage needs at least one layer."  # noqa: B905
        assert not (downsample_only and return_down_path), "Without a decoder the down-path features are the output features."
        assert not deepest_mid_target or return_down_path, "The mid target of the deepest stage is returned with the down path."
        assert not deepest_mid_target or (len(stage_dims) - 1) in out_norm_stages, "The deepest stage must be supervised."
        assert not deepest_mid_target or num_down_layers_per_stage[-1] >= 2, "The deepest stage needs at least two layers to have a middle."
        super().__init__()
        self._downsample_only = downsample_only
        self._out_norm_stages = out_norm_stages
        # Up layers per stage, 0 for the deepest one, and where each stage's up layers start in _us_attn_layers.
        self._num_up_layers = [*num_up_layers_per_stage, 0]
        self._up_layer_starts = [sum(self._num_up_layers[:s]) for s in range(len(stage_dims))]
        # Number of layers of the deepest stage after which its mid target is taken, None without one.
        self._deepest_mid_layer = num_down_layers_per_stage[-1] // 2 if deepest_mid_target else None
        self._has_registers = num_registers > 0

        if condition_voxel_size:
            # Per stage, shared by all its layers: u -> [u, sin(f*u), cos(f*u)] -> Linear(17, 64) -> GELU -> Linear(64, 6*C)
            # -> (B, 6, C) deltas of (alpha_attn, beta_attn, gamma_attn, alpha_ffn, beta_ffn, gamma_ffn). Each layer adds its own _adaln_vs_bias.
            # Equivalent to doing embed then adaln projection. 
            self._voxel_size_embed = nn.ModuleList(
                [VoxelSizeEmbedding(6 * sd, out_init_std=voxel_size_cond_init_std) for sd in stage_dims]
            )
        else:
            self._voxel_size_embed = None
        self._voxel_size_range = voxel_size_range

        if self._has_registers:
            self._register_init = nn.Parameter(torch.randn(num_registers, stage_dims[0]))
            self._reg_pool_proj = nn.ModuleList([nn.Linear(s_low, s_high, bias=True) for s_low, s_high in zip(stage_dims[:-1], stage_dims[1:])])  # noqa: B905
            if self._downsample_only:
                self._reg_unpool_proj = None
            else:
                self._reg_unpool_proj = nn.ModuleList([nn.Linear(stage_dims[i + 1], stage_dims[i], bias=True) 
                                                    for i in range(len(stage_dims) - 1) if i in self._out_norm_stages])  # noqa: B905
                
        self._voxel_size_point_feature = voxel_size_point_feature
        # Lets add a initial projection layer for the input points so that the LN in the initial pooling layer does 
        # not destroy the point information of smaller scale (coordinates vs reflectivity scale).
        in_dim = POINT_FEATURE_DIM + INPUT_PE_DIM + int(voxel_size_point_feature)
        self._init_pe = PEDynamic(embed_dim=INPUT_PE_DIM, pos_dim=3, max_freq=1000, perceiver=False)
        self._init_proj = nn.Sequential(nn.Linear(in_dim, stage_dims[0]), nn.GELU())
        self._init_pool = Pool(stride=2, in_dim=stage_dims[0], out_dim=stage_dims[0], activation=nn.GELU(), postnorm=pool_postnorm)
        self._group_size = group_size

        # self._ropes = nn.ModuleList([RoPE(dim=dim, pos_dim=3) for dim in stage_dims])
        self._ds_ropes = nn.ModuleList([])
        self._ds_attn_groups = nn.ModuleList([])

        for nl, sd, hd in zip(num_down_layers_per_stage, stage_dims, num_heads_per_stage):  # noqa: B905
            self._ds_attn_groups.append(
                nn.ModuleList(
                    [
                        PointSelfAttn(
                            dim=sd,
                            num_heads=hd,
                            expansion_factor=4,
                            glu=False,
                            group_size=group_size,
                            weight_mix_qk=0.95,
                            register_reduction=register_reduction,
                            gradient_checkpointing=False,
                            sep_reg_voxel_norms=sep_reg_voxel_norms,
                            use_attention_bias=use_attention_bias,
                            rope_layer=RoPE(dim=sd, pos_dim=3, learnable_scale=False, max_freq=100, num_heads=hd),
                            qk_layernorm=qk_layernorm,
                            condition_voxel_size=condition_voxel_size,
                            xsa=xsa,
                            sdpa_gate=sdpa_gate,
                            qkv_bias=qkv_bias,
                            qk_layernorm_impl=qk_layernorm_impl,
                        )
                        for _ in range(nl)
                    ]
                )
            )

        # TODO The DDP order is inefficient since everything is defined out of order!!!
        self._pool_layers = nn.ModuleList([Pool(stride=2, in_dim=stage_dims[i], out_dim=stage_dims[i + 1], postnorm=pool_postnorm) for i in range(len(stage_dims) - 1)])
        if not downsample_only:
            # only need upsampling related layers for stages that we apply a loss on (out_norm_stages).
            # The up layers of all stages are in one list, shallowest stage first, so that with one up layer per
            # stage the parameter names are those of the former single up layer per stage.
            self._us_attn_layers = nn.ModuleList(
                [
                    PointSelfAttn(
                        dim=sd,
                        num_heads=hd,
                        expansion_factor=4,
                        glu=False,
                        group_size=group_size,
                        weight_mix_qk=0.95,
                        gradient_checkpointing=False,
                        sep_reg_voxel_norms=sep_reg_voxel_norms,
                        use_attention_bias=use_attention_bias,
                        rope_layer=RoPE(dim=sd, pos_dim=3, learnable_scale=False, max_freq=100, num_heads=hd),
                        qk_layernorm=qk_layernorm,
                        condition_voxel_size=condition_voxel_size,
                        xsa=xsa,
                        sdpa_gate=sdpa_gate,
                        qkv_bias=qkv_bias,
                        qk_layernorm_impl=qk_layernorm_impl,
                    )
                    for sd, hd, nl in zip(stage_dims[:-1], num_heads_per_stage[:-1], num_up_layers_per_stage)  # noqa: B905
                    for _ in range(nl)
                ]
            )

            self._unpool_layers = nn.ModuleList(
                [UnPool(dim_lowres=stage_dims[i + 1], dim_highres=stage_dims[i]) for i in range(len(stage_dims) - 1) if i in self._out_norm_stages]
            )

        self._final_layernorms = nn.ModuleList([nn.LayerNorm(stage_dims[s]) for s in out_norm_stages])
        # Not shared with the final norms since the down-path features have different statistics. The norm of the
        # deepest stage's mid target comes last, so these keep their indices with and without it.
        self._down_layernorms = (
            nn.ModuleList(
                [nn.LayerNorm(stage_dims[s]) for s in out_norm_stages if s < len(stage_dims) - 1]
                + ([nn.LayerNorm(stage_dims[-1])] if deepest_mid_target else [])
            )
            if return_down_path
            else None
        )
        
        self._group_size = group_size
        self._stage_dims = stage_dims
        self._voxel_size = voxel_size
        self._pos_norm_voxel_size = pos_norm_voxel_size

    @property
    def num_stages(self):
        return len(self._stage_dims)

    @property
    def voxel_size(self):
        return self._voxel_size

    def stage_voxel_sizes(self, voxel_size: float | torch.Tensor) -> list[float | torch.Tensor]:
        """Scalar or (B,) input voxel sizes -> voxel sizes at each attention stage, keeping their type."""
        voxel_size_init = voxel_size * self._init_pool._stride
        return [voxel_size_init * (2**s) for s in range(self.num_stages)]

    def _stem_forward(
        self,
        ptcl: torch.Tensor,
        batch_idx: torch.Tensor,
        grid_idx: torch.Tensor | None,
        batch_size: int | None,
        voxel_size: float | torch.Tensor,
        pos_voxel_size: float | torch.Tensor,
        voxel_size_u: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Embed, serialize and pool raw points inside one checkpoint boundary in forward."""
        pos_s0 = ptcl[..., XYZ_SLICE]
        pos_divisor = pos_voxel_size[batch_idx].unsqueeze(-1) if isinstance(pos_voxel_size, torch.Tensor) else pos_voxel_size
        # Raw point features and a PE of the position in voxels.
        feat_s0 = [ptcl, self._init_pe(pos_s0 / pos_divisor)]
        if self._voxel_size_point_feature:
            feat_s0.append(voxel_size_u[batch_idx].to(ptcl.dtype).view(-1, 1))
        feat_s0 = torch.cat(feat_s0, dim=-1)

        if grid_idx is None:
            grid_divisor = voxel_size[batch_idx] if isinstance(voxel_size, torch.Tensor) else voxel_size
            grid_idx, _ = get_grid_idx(pos_s0, batch_idx, grid_divisor, batch_size=batch_size)
        sf_idx = space_filling_idx(grid_idx)
        feat_s0 = self._init_proj(feat_s0)
        return self._init_pool(feat_s0, pos_s0, sf_idx, batch_idx, batch_size)

    @torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    def forward(
        self, ptcl, batch_idx, grid_idx = None, batch_size: int | None = None, return_down_path: bool = False, voxel_size: torch.Tensor | None = None,
    ):
        """Forward pass of the point transformer.

        Raw-point embedding, serialization and initial pooling share one non-reentrant checkpoint,
        so unpooled feature tensors are recomputed during backward instead of retained.

        Args:
            ptcl: (N, POINT_FEATURE_DIM), [x, y, z, intensity] with optional leading time
            batch_idx: (N) batch indices for each point.
            grid_idx: (N, 3) precomputed grid indices for each point. This can optionally be used to
                      align the grids between different views of the same data (teacher and student).
                      Computed on-the-fly if not provided.
            return_down_path: Additionally return the normalized down-path features of each stage.
            voxel_size: (B,) voxel size of each sample. Defaults to the voxel size given at construction.
                        Fixed default/reference scales use Python floats to preserve historical CUDA rounding.
                        Explicit per-sample voxel sizes remain tensors.
                        NOTE/WARNING: Must match the voxel size used to compute grid_idx if that is given!!

        Returns:
            Lists of:
            features and positions of points at each stage along with upsample indices to go from
            lower resolution stage to higher resolution stage.
            At the last stage (last idx in up_idx list) upsample indices are None.
            With return_down_path, also the down-path features of each stage, which are None for
            stages without a decoder (including the last one unless it has a mid target, which is returned there).
        """
        assert batch_idx.shape[0] == ptcl.shape[0]
        assert (grid_idx is None) or (grid_idx.shape[0] == ptcl.shape[0])
        assert not return_down_path or self._down_layernorms is not None, "Encoder was built without down-path norms."
        pt_feats = []
        pt_positions = []
        batch_indices = []
        up_idxs = []
        orders = []
        inv_orders = []
        paddings = []
        unpaddings = []
        sf_curve_idxs = []

        B = batch_size if batch_size is not None else batch_idx.max().item() + 1
        grid_voxel_size = self._voxel_size if voxel_size is None else voxel_size
        if voxel_size is None:
            voxel_size = torch.full((B,), self._voxel_size, device=ptcl.device, dtype=torch.float32)
        assert voxel_size.shape == (B,)
        pos_voxel_size = grid_voxel_size if self._pos_norm_voxel_size is None else self._pos_norm_voxel_size
        pos_stage_voxel_sizes = self.stage_voxel_sizes(pos_voxel_size)
        voxel_size_u = log_voxel_size_coordinate(voxel_size, self._voxel_size, self._voxel_size_range)

        if self._voxel_size_embed is not None:
            with torch.autocast("cuda", enabled=False):
                # (B, 6, C) per stage, shared by all attention layers of the stage.
                # Using input voxelsize across stages is fine because the embedding can absorb the differences in voxel size for each stage.
                voxel_size_mod_stages = [
                    embed(voxel_size_u).view(B, 6, -1) for embed in self._voxel_size_embed
                ]
        else:
            voxel_size_mod_stages = [None] * self.num_stages

        feat_stage, pos_stage, batch_idx_stage, sf_idx, order, inv_order, init_up_idx = checkpoint(
            self._stem_forward,
            ptcl,
            batch_idx,
            grid_idx,
            batch_size,
            grid_voxel_size,
            pos_voxel_size,
            voxel_size_u,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        up_idxs.append(init_up_idx)

        registers = []
        current_register = self._register_init.expand(B, *self._register_init.shape) if self._has_registers else None
        for stage in range(self.num_stages):
            # for each stage start at point features and positions of prev stage
            # -> add RoPE (pos enc)
            # -> pad to multiple of group size -> apply attn layers in _ds_attn_groups[stage]
            # -> if not last stage pool to next stage resolution using pool_layers[stage]
            if stage > 0 and current_register is not None:
                current_register = self._reg_pool_proj[stage - 1](current_register)

            pad, unpad = pad_and_unpad_idx(
                batch_indices=batch_idx_stage,
                group_size=self._group_size,
                batch_size=batch_size,
            )

            # Indices we will use for keeping track of which space-filling curve to use.
            # Intentionally on the cpu. The up layers of this stage continue the sequence after the down layers.
            num_curve_layers = len(self._ds_attn_groups[stage]) + self._num_up_layers[stage]
            sf_curve_idx = torch.cat([torch.randperm(4) for _ in range(math.ceil(num_curve_layers / 4))], dim=0)
            sf_curve_idxs.append(sf_curve_idx)
            for i, layer in enumerate(self._ds_attn_groups[stage]):
                order_idx = sf_curve_idx[i] if self.training else i % 4
                pos_divisor = pos_stage_voxel_sizes[stage]
                if isinstance(pos_divisor, torch.Tensor):
                    pos_divisor = pos_divisor[batch_idx_stage].unsqueeze(-1)
                feat_stage, current_register = checkpoint(
                    layer,
                    feat_stage,
                    order[order_idx],
                    inv_order[order_idx],
                    pad,
                    unpad,
                    pos_stage / pos_divisor,
                    current_register,
                    batch_idx_stage,
                    voxel_size_mod_stages[stage],
                    batch_size=B,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                if stage == self.num_stages - 1 and i + 1 == self._deepest_mid_layer:
                    deepest_mid_feat = feat_stage

            pt_feats.append(feat_stage)
            pt_positions.append(pos_stage)
            batch_indices.append(batch_idx_stage)
            if current_register is not None:
                registers.append(current_register)

            if stage < (self.num_stages - 1):
                # Saving order and inverse order must be done before pooling but
                # only needed for upsampling (so not for last stage)
                orders.append(order)
                inv_orders.append(inv_order)
                # Save padding and unpadding too (also need for upsampling)
                paddings.append(pad)
                unpaddings.append(unpad)

                # Pool to next stage resolution and get new space-filling curves
                feat_stage, pos_stage, batch_idx_stage, sf_idx, order, inv_order, up_idx = checkpoint(
                    self._pool_layers[stage],
                    feat_stage,
                    pos_stage,
                    sf_idx,
                    batch_idx_stage,
                    batch_size,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                up_idxs.append(up_idx)

        if return_down_path:
            # Must happen before the decoder overwrites pt_feats.
            down_stages = [s for s in self._out_norm_stages if s < self.num_stages - 1]
            down_feats = [None] * self.num_stages
            for norm, s in zip(self._down_layernorms[: len(down_stages)], down_stages, strict=True):
                down_feats[s] = norm(pt_feats[s])
            if self._deepest_mid_layer is not None:
                down_feats[-1] = self._down_layernorms[-1](deepest_mid_feat)
            
        if not self._downsample_only:
            # Do the upsampling
            current_register = registers[-1] if registers else None
            for stage in reversed(range(self.num_stages - 1)):
                if stage in self._out_norm_stages:
                    idx = self._out_norm_stages.index(stage)
                    # Upsample from lower resolution stage to current stage using unpool_layers[stage]
                    # then apply the up layers of this stage in _us_attn_layers
                    current_register = None if current_register is None else self._reg_unpool_proj[idx](current_register)
                    feat_stage = checkpoint(
                        self._unpool_layers[idx],
                        pt_feats[stage + 1],
                        pt_feats[stage],
                        up_idxs[stage + 1],
                        use_reentrant=False,
                        preserve_rng_state=False
                    )
                    start = self._up_layer_starts[stage]
                    up_layers = self._us_attn_layers[start : start + self._num_up_layers[stage]]
                    # i counts on from the down layers through the curve sequence of the stage.
                    for i, layer in enumerate(up_layers, start=len(self._ds_attn_groups[stage])):
                        order_idx = sf_curve_idxs[stage][i] if self.training else i % 4
                        pos_divisor = pos_stage_voxel_sizes[stage]
                        if isinstance(pos_divisor, torch.Tensor):
                            pos_divisor = pos_divisor[batch_indices[stage]].unsqueeze(-1)
                        feat_stage, current_register = checkpoint(
                            layer,
                            feat_stage,
                            orders[stage][order_idx],
                            inv_orders[stage][order_idx],
                            paddings[stage],
                            unpaddings[stage],
                            pt_positions[stage] / pos_divisor,
                            current_register,
                            batch_indices[stage],
                            voxel_size_mod_stages[stage],
                            batch_size=B,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    if current_register is not None:
                        registers[stage] = current_register

                    pt_feats[stage] = feat_stage

        norm_count = 0
        for s, feat in enumerate(pt_feats):
            if s in self._out_norm_stages:
                feat = self._final_layernorms[norm_count](feat)
                pt_feats[s] = feat
                norm_count += 1

        registers = registers if self._has_registers else None # None instead of empty list if we don't have registers
        if return_down_path:
            return pt_feats, pt_positions, batch_indices, up_idxs, registers, down_feats
        return pt_feats, pt_positions, batch_indices, up_idxs, registers


if __name__ == "__main__":
    batch_indices = torch.tensor([0, 0, 0, 0, 0])
    pad, unpad = pad_and_unpad_idx(batch_indices, 2)
    print("We expect:")
    # print("pad = [0, 1, 0, 1, 0, 1], unpad = [0, 1")
    print(f"{pad=}")
    print(f"{unpad=}")

    # x = torch.randn(1001, 5)
    # ptv = PointTransformer(group_size=100)
    # out = ptv(x)
    # for o in out:
    #    print(o.shape)
