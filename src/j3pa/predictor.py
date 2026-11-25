"""Predictor for point jepa"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from j3pa.positional_encodings import PEDynamic
from j3pa.pt_transf import PointSelfAttn, batched_permute, pad_and_unpad_idx, space_filling_idx, get_grid_idx
from j3pa.rope import RoPE
from j3pa.layers import FiLM
from constants import USE_REGISTERS_IN_PREDICTOR

class PointCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion_factor: int,
        group_size: int,
        zero_residual=False,
        weight_tie_qk=False,
        use_mask_size_film=False,
        mask_size_embed_dim=None
    ):
        assert dim % num_heads == 0
        if use_mask_size_film:
            assert mask_size_embed_dim is not None
        else:
            assert mask_size_embed_dim is None
        super().__init__()
        self._q_ln_attn = nn.LayerNorm(dim)
        self._kv_ln_attn = nn.LayerNorm(dim)
        self._pq = nn.Linear(dim, dim, bias=False)

        self._pk = nn.Linear(dim, dim, bias=False)
        self._pv = nn.Linear(dim, dim, bias=False)
        # Symmetry in attentions nonlinearity (softmax) makes Glorot initialization preferrable to
        # make QKV projections. We can keep Kaiming in the ffn.
        nn.init.xavier_uniform_(self._pq.weight)
        nn.init.xavier_uniform_(self._pk.weight)
        nn.init.xavier_uniform_(self._pv.weight)

        self._po = nn.Linear(dim, dim)
        self._num_heads = num_heads
        self._group_size = group_size

        self._attn_ms_film = FiLM(mask_size_embed_dim, dim, use_encoder_in_condition=False) if use_mask_size_film else None

        self._ln_ffn = nn.LayerNorm(dim)
        self._ffn = nn.Sequential(
            nn.Linear(dim, expansion_factor * dim, bias=(expansion_factor > 1)),
            nn.GELU(),
            nn.Linear(expansion_factor * dim, dim),
        )
        self._ffn_ms_film = FiLM(mask_size_embed_dim, dim, use_encoder_in_condition=False) if use_mask_size_film else None
        
        if weight_tie_qk:
            self._pq.weight.data = self._pk.weight.data.clone()

        if zero_residual:
            # Start with clean residuals like modern LLM trainings.
            nn.init.zeros_(self._po.weight)
            nn.init.zeros_(self._po.bias)
            nn.init.zeros_(self._ffn[-1].weight)
            nn.init.zeros_(self._ffn[-1].bias)
        else:
            nn.init.xavier_uniform_(self._po.weight)
            nn.init.zeros_(self._po.bias)

    def forward(self, pt_feat, q_feat, pt_pos, q_pos, order_idx, rope: RoPE = None, register = None, query_batch_indices = None, mask_size_embed = None):
        """Args:
        pt_feat: (K, C)
        q_feat: (Q, C)
        order_idx: (Q*X) where X is group size
        inv_order_idx: (Q*X)

        Each query shall attend to X points.
        """
        assert order_idx.shape[0] % self._group_size == 0
        if register is not None:
            assert len(register.shape) == 3
            assert register.shape[-1] == pt_feat.shape[-1]
            assert query_batch_indices is not None
            assert query_batch_indices.shape[0] == q_feat.shape[0]
        if self._attn_ms_film is not None:
            assert mask_size_embed is not None

        num_groups = order_idx.shape[0] // self._group_size
        Q, _ = q_feat.shape
        assert num_groups == Q
        K, C = pt_feat.shape
        q = self._pq(self._q_ln_attn(q_feat))  # (Q, C)
        normed_kv = self._kv_ln_attn(pt_feat)
        k = self._pk(normed_kv)  # (K, C)
        v = self._pv(normed_kv)  # (K, C)
        del normed_kv
        if rope:
            rope.cache_cos_sin(q_feat, q_pos)
            q = rope.apply_precomputed(q)
            rope.cache_cos_sin(pt_feat, pt_pos)
            k = rope.apply_precomputed(k)

            # go to (num_groups, num_heads, num_points_in_group, channel_size_in_head)
            # = (Q, H, 1, C//H) for queries and = (Q, H, X, C//H) for keys and values

        q = q.view(num_groups, 1, self._num_heads, C // self._num_heads).permute(0, 2, 1, 3)  # (Q, H, 1, C//H)

        # (K C) -> (Q*X, C) ->  (Q, X, H, C//H) -> (Q, H, X, C//H)
        k = k[order_idx].view(num_groups, self._group_size, self._num_heads, C // self._num_heads).permute(0, 2, 1, 3)
        v = v[order_idx].view(num_groups, self._group_size, self._num_heads, C // self._num_heads).permute(0, 2, 1, 3)

        if register is not None:
            B, R, C = register.shape
            normed_reg = self._kv_ln_attn(register)

            # (B, R, C) -> (B, R, H, C//H) -> (B, H, R, C//H)
            rk = self._pk(normed_reg).view(B, R, self._num_heads, C // self._num_heads).permute(0, 2, 1, 3)
            rv = self._pv(normed_reg).view(B, R, self._num_heads, C // self._num_heads).permute(0, 2, 1, 3)

            # Use queries batch indices to select the correct registers for each query group. 
            rk = rk[query_batch_indices] # (Q, H, R, C//H)
            rv = rv[query_batch_indices] # (Q, H, R, C//H)

            k = torch.cat([k, rk], dim=-2) # (Q, H, X+R, C//H)
            v = torch.cat([v, rv], dim=-2) # (Q, H, X+R, C//H)

        # (Q, H, 1, C//H) hopefully or i have messed     up
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # (Q, H, 1, C//H) -> (Q, C)
        attn_out = attn_out.permute(0, 2, 1, 3).view(num_groups, C)
        attn_out = self._po(attn_out)

        if self._attn_ms_film is not None:
            q_feat = q_feat + self._attn_ms_film(mask_size_embed, attn_out, query_batch_indices) 
        else:
            q_feat = q_feat + attn_out

        if self._ffn_ms_film is not None:
            return q_feat + self._ffn_ms_film(mask_size_embed, self._ffn(self._ln_ffn(q_feat)), query_batch_indices)
        return q_feat + self._ffn(self._ln_ffn(q_feat))
        


def group_and_order_idx(sf_idx: torch.Tensor, key_count: int, key_batch_indices: torch.Tensor, query_batch_indices: torch.Tensor, group_size: int):
    """Provide the order indices to go go from K points to Q*X.

    Where Q is number of queries and X is group size. with each group containing the
    X closest points to the query in the space-filling curve ordering.

    Args:
        sf_idx: (S, K+Q) space-filling curve indices for all points where the first K are the keys and the last Q are the queries.
        key_count: Number of keys (K)

    where S is number of space-filling curves, K is number of keys and Q is number of queries.

    Returns:
        order: (S, Q*group_size) indices to go from K to Q*group_size.
    """
    S = sf_idx.shape[0]
    K = key_count
    Q = sf_idx.shape[1] - K

    sf_idx_keys = sf_idx[:, :K]  # (S, K)
    sf_idx_queries = sf_idx[:, K:]  # (S, Q)

    half_group = group_size // 2
    offsets = torch.arange(group_size, device=sf_idx_keys.device) - half_group

    # Make linear sf indices across batches
    # Important to offset by 1 when making linear indices.
    max_sf_idx = max(sf_idx_keys.amax(), sf_idx_queries.amax()) + 1

    # (S, K)
    linear_sf_idx_keys = key_batch_indices * max_sf_idx + sf_idx_keys

    # (S, Q)
    linear_sf_idx_queries = query_batch_indices * max_sf_idx + sf_idx_queries

    linear_sorted, linear_sort_idx = torch.sort(linear_sf_idx_keys, dim=1)  # (S, K)
    insert_pos = torch.searchsorted(linear_sorted.contiguous(), linear_sf_idx_queries.contiguous())  # (S, Q)

    # We need to go to (S, Q, group_size). For each query referring to indices of different keys.
    windows = insert_pos.unsqueeze(2) + offsets  # (S, Q, group_size)

    # Now find the minimum and maximum index in each batch for each query (Q)
    # count entries per batch
    one_k = torch.ones_like(key_batch_indices).int()
    counts = torch.empty((key_batch_indices.max().item() + 1), device=one_k.device, dtype=torch.int32)
    counts.scatter_reduce_(0, key_batch_indices, one_k, reduce="sum", include_self=False)

    cumalative_counts = torch.cumsum(counts, 0)
    min_offsets = torch.cat([torch.zeros_like(counts[0:1]), cumalative_counts[:-1]])[query_batch_indices]  # B -> Q
    max_offsets = cumalative_counts[query_batch_indices]

    min_offsets = min_offsets.view(1, -1, 1)
    max_offsets = max_offsets.view(1, -1, 1)

    windows = torch.where(windows >= min_offsets, windows, windows + group_size)  # (S, Q, group_size)
    num_points_in_batch = torch.clamp(
        max_offsets - min_offsets, min=1
    )  # least num in a batch that gets included here is 1 als mod 1 = 0 so it works.
    # now since line above the indices in windows are always respecting lower bound but not upper necessarily...
    windows = torch.where(
        windows < max_offsets, windows, torch.max((windows - group_size), min_offsets + windows % num_points_in_batch)
    )  # (S, Q, group_size)

    S_idx = torch.arange(S, device=sf_idx_keys.device).view(S, 1, 1)
    order = linear_sort_idx[S_idx, windows].view(S, Q * group_size)  # (S, Q*group_size)
    return order


class Predictor(nn.Module):
    # TODO: Experiment with group size
    def __init__(
        self,
        encoder_stage_dims: list,
        predictor_stage_dims: list,
        voxel_size: float,
        group_size: int = 256,
        init_pe_dim: int | None = None,
        layers_per_stage: int = 3,
        use_cross_attn: bool = True,
        use_mask_size_film: bool = False
    ):
        """
        Args:
            encoder_stage_dims: List of feature dimensions at each stage of the encoder.
            predictor_stage_dims: List of feature dimensions at each stage of the predictor.
                                In I-JEPA, V-JEPA, and V-JEPA 2 the predictor always has only 384
                                channels, regardless of the size of the encoder.
            voxel_size: Voxel size.
            group_size: How many embeddings each query attends to.
            init_pe_dim: Initial positional encoding dimension if provided.
            layers_per_stage: Number of attention layers per stage in the predictor.
            use_cross_attn: Whether to use cross attention between student and teacher features.
                            If False, the predictor will only use self-attention on concatenated
                            student and teacher features. If True each query point will attend to
                            a group of student points in its neighborhood independently.
            use_mask_size_film: Whether to condition the predictor on the mask sizes or not.
        """
        super().__init__()
        assert len(encoder_stage_dims) == len(predictor_stage_dims), (
            f"Expect the same number of encoder and corresponding predictor stages, "
            f"but got {len(encoder_stage_dims)} and {len(predictor_stage_dims)}."
        )
        assert all(d % 16 == 0 for d in predictor_stage_dims), (
            f"We divide attention heads per 16 channels but got predictor dimensionalities of {predictor_stage_dims}."
        )

        self._stage_dims = predictor_stage_dims
        self._num_stages = len(predictor_stage_dims)

        # Define input projections before attention for more streamlined DDP communication structure
        # No layernorm is added on top of this in I-JEPA and V-JEPA. I suppose so that the whole
        # structure stays as close to prenorm as possible.
        self._context_projs = nn.ModuleList([nn.Linear(ed, pd, bias=True) for ed, pd in zip(encoder_stage_dims, predictor_stage_dims, strict=False)])
        
        self._use_registers = USE_REGISTERS_IN_PREDICTOR
        self._reg_projs = nn.ModuleList([nn.Linear(ed, pd, bias=True) for ed, pd in zip(encoder_stage_dims, predictor_stage_dims, strict=False)]) if self._use_registers else None

        # How to embed mask token positions best way????  TODO
        in_dim = 3 + init_pe_dim if init_pe_dim else 3
        self._mask_pos_projs = nn.ModuleList([nn.Linear(in_dim, pd) for pd in predictor_stage_dims])

        self._attn_blocks = nn.ModuleList([])
        self._ropes = nn.ModuleList([])
        self._use_mask_size_film = use_mask_size_film

        # TODO: Add the ability for modulate based on the current latent.
        for pd, ed in zip(predictor_stage_dims, encoder_stage_dims):
            self._attn_blocks.append(
                nn.ModuleList(
                    [
                        PointCrossAttn(
                            dim=pd,
                            # TODO Experiment with the head size.
                            num_heads=pd // 16,
                            expansion_factor=4,
                            group_size=group_size,
                            use_mask_size_film=use_mask_size_film,
                            mask_size_embed_dim=ed if use_mask_size_film else None
                        ) if use_cross_attn else PointSelfAttn(
                            dim=pd,
                            num_heads=pd // 16,
                            expansion_factor=4,
                            group_size=group_size,
                        )
                        for _ in range(layers_per_stage)
                    ]
                )
            )
            self._ropes.append(
                # Each attention block can tune its own basis angle.
                nn.ModuleList([RoPE(dim=pd, pos_dim=3, max_freq=1e4, learnable_scale=False) for _ in range(layers_per_stage)])
            )

        self._group_size = group_size
        self._voxel_size = voxel_size
        self._use_cross_attn = use_cross_attn

        self._init_pe = PEDynamic(embed_dim=init_pe_dim, pos_dim=3, max_freq=1e4, perceiver=False) if init_pe_dim else None
        # Dumb, bias can absorb this. Need if we do other thing like fourier encoding or whatever
        # self._mask_tokens = nn.ModuleList([nn.Parameter(torch.randn(sd)) for sd in stage_dims])

        self._out_projs = nn.ModuleList(
            # Note that if our predictor is narrower than the encoder (not the case yet) the bias
            # can not be absorbed by the affine norm.
            [nn.Sequential(nn.LayerNorm(pd), nn.Linear(pd, ed, bias=True)) for pd, ed in zip(predictor_stage_dims, encoder_stage_dims, strict=False)]
        )

    @property
    def stage_dims(self):
        return self._stage_dims

    @property
    def num_stages(self):
        return self._num_stages

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def use_mask_size_film(self):
        return self._use_mask_size_film

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(
        self,
        student_feat_stages: list[torch.Tensor],
        student_pos_stages: list[torch.Tensor],
        masked_teacher_pos_stages: list[torch.Tensor],
        student_batch_indices_stages: list[torch.Tensor],
        masked_teacher_batch_indices_stages: list[torch.Tensor],
        register_tokens: list[torch.Tensor] | None = None,
        mask_size_embeddings: list[torch.Tensor] | None = None,
    ):
        """Forward pass of the predictor.
        Args:
            student_feat_stages: List of student features at each stage. Each element is (N, C).
            student_pos_stages: List of student positions at each stage. Each element is (N, 3).
            masked_teacher_pos_stages: List of masked teacher positions at each stage. Each element is (M, 3)
            student_batch_indices_stages: List of batch indices for student points at each stage. Each element is (N,)
            masked_teacher_batch_indices_stages: List of batch indices for masked teacher points at each stage. Each element is (M,)
            mask_size_embeddings: List of mask embeddings in case we want to do conditioning on the mask size.

            N and M are the concatinated points and masked points in the batch respectively.
        Returns:
            List of predicted teacher features at each stage. Each element is (M, C) where M is the number of
            masked points at that stage and C is the corresponding encoder stage dimension.
        """
        assert len(student_feat_stages) == len(masked_teacher_pos_stages)
        assert len(student_feat_stages) == len(student_pos_stages)
        assert len(self._attn_blocks) == len(student_pos_stages)
        if self._use_mask_size_film:
            assert mask_size_embeddings is not None
            assert len(mask_size_embeddings) == len(student_pos_stages)
        outs = []
        
        for (
            student_feat,
            student_pos,
            teacher_pos,
            student_bi,
            teacher_bi,
            reg_tokens,
            mask_size_embed,
            attn_blocks,
            ropes,
            context_proj,
            masked_proj,
            out_proj,
            reg_proj,
        ) in zip(
            student_feat_stages,
            student_pos_stages,
            masked_teacher_pos_stages,
            student_batch_indices_stages,
            masked_teacher_batch_indices_stages,
            register_tokens if self._use_registers else [None] * len(student_feat_stages),
            mask_size_embeddings if self._use_mask_size_film else [None] * len(student_feat_stages),
            self._attn_blocks,
            self._ropes,
            self._context_projs,
            self._mask_pos_projs,
            self._out_projs,
            self._reg_projs if self._use_registers else [None] * len(self._attn_blocks),
            strict=False,
        ):
            assert teacher_pos.shape[0] > 0, f"No masked points at this stage when encoder dim = {student_feat.shape[1]} Should not happen"

            context_feats = context_proj(student_feat)
            reg_feats = reg_proj(reg_tokens) if reg_tokens is not None else None
            if self._init_pe is not None:
                q_feat = masked_proj(torch.cat([teacher_pos, self._init_pe(teacher_pos)], dim=-1))
            else:
                q_feat = masked_proj(teacher_pos)

            all_pos = torch.cat([student_pos, teacher_pos], dim=0)
            # (4, N)
            # TODO If we have more than 4 attention layers in a block, randperm again!
            all_bi = torch.cat([student_bi, teacher_bi], dim=0)
            grid_idx, _ = get_grid_idx(all_pos, all_bi, self._voxel_size)
            sf_idx = space_filling_idx(grid_idx)
            sf_idx = batched_permute(sf_idx, all_bi)
            if self._use_cross_attn:
                order = group_and_order_idx(sf_idx, student_pos.shape[0], student_bi, teacher_bi, self._group_size)
            else:
                # (4, N)
                order = torch.argsort(sf_idx, dim=-1)
                inv_order = torch.empty_like(order)
                inv_order.scatter_(
                    -1,
                    order,
                    torch.arange(order.size(-1), device=order.device).expand_as(order),
                )

                pad, unpad = pad_and_unpad_idx(
                    num_points=all_pos.shape[0],
                    group_size=self._group_size,
                    device=student_feat.device,
                )
                all_feat = torch.cat([context_feats, q_feat], dim=0)

            for i, (attn, rope) in enumerate(zip(attn_blocks, ropes, strict=False)):
                order_idx = i % 4
                if self._use_cross_attn:
                    q_feat = checkpoint(
                        attn,
                        context_feats,
                        q_feat,
                        student_pos,
                        teacher_pos,
                        order[order_idx],
                        rope,
                        reg_feats,
                        teacher_bi,
                        mask_size_embed=mask_size_embed,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    out = q_feat
                else:
                    rope.cache_cos_sin(all_feat, all_pos)
                    all_feat = attn(
                        all_feat,
                        order[order_idx],
                        inv_order[order_idx],
                        pad,
                        unpad,
                        rope,
                        reg_feats,
                        all_bi,
                        mask_size_embed=mask_size_embed,
                    )
                    out = all_feat[student_feat.shape[0] :]

            outs.append(out_proj(out))

            del order

        return outs


if __name__ == "__main__":
    predictor = Predictor([64], [64], 1, group_size=2)
    st_feat = [torch.randn(5000, 64), torch.randn(1000, 128)]
    st_pos = [torch.randn(5000, 3), torch.randn(1000, 3)]
    mt_pos = [torch.randn(400, 3), torch.randn(200, 3)]
    st_feat = [torch.randn(7, 64)]
    st_pos = [torch.tensor([[1, 1, 1], [1, 1, 0], [2, 2, 2], [5, 3, 4], [0, 0, 0], [1, 1, 0], [3, 3, 3]], dtype=torch.float)]
    mt_pos = [torch.tensor([[1, 0, 0.5], [2.25, 2, 2], [2, 2, 2]], dtype=torch.float)]
    batch_indices_st = [torch.tensor([0, 0, 0, 0, 1, 1, 1])]
    batch_indices_mt = [torch.tensor([0, 1, 0])]
    out = predictor(st_feat, st_pos, mt_pos, batch_indices_st, batch_indices_mt)

    for o in out:
        print(o.shape)
