"""Deformable predictor for point jepa.

Per-offset softmax merge of the K neighbours on metric distances, softmax over the offsets, no value PE, and queries
initialised from their raw (metric) positions.
"""

from typing import Literal, NamedTuple

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from j3pa.positional_encodings import PEDynamic
from j3pa.layers import FiLM, VoxelSizeEmbedding, log_voxel_size_coordinate
from pytorch3d.ops import knn_points


def knn_indices_bruteforce(q_pos, q_bi, c_pos, c_bi, K):
    """Our old Knn indices implementation.
    
    This one turns different batch indices into isolated "islands" in 4D space so that the knn indices always come from
    the same batch.
    """
    max_dist = torch.max(q_pos.amax(0) - c_pos.amin(0), c_pos.amax(0) - q_pos.amin(0)).norm()
    batch_scale = max_dist*1.01 + 1e-3
    q_4th = (batch_scale * q_bi).unsqueeze(-1)
    pad_q= torch.cat([q_pos, q_4th], dim=-1)
    c_4th = (batch_scale * c_bi).unsqueeze(-1)
    pad_c = torch.cat([c_pos, c_4th], dim=-1)
    _, knn_idx, _ = knn_points(pad_q.unsqueeze(0), pad_c.unsqueeze(0), norm=2, K=K, return_nn=False, return_sorted=True)
    knn_idx = knn_idx.squeeze(0)
    return knn_idx


class _RaggedPtcl(NamedTuple):
    """Helper container for pytorch3d ragged point clouds.
    
    The output is batch_sorted and contains the padded point cloud, the number of entries in each
    sample, the start indices in of each point in the original tensor and the 
    """
    ptcl_padded: torch.Tensor  # (B, max_n, D)
    num_pts: torch.Tensor  # (B,) real points per sample
    starts: torch.Tensor  # (B,) flat index of each sample's first point in the original tensor.
    in_sample_idx: torch.Tensor  # (N,) Index inside sample for each point.


def _to_ragged(pos: torch.Tensor, bi: torch.Tensor, B: int) -> _RaggedPtcl:
    """Go from batch-sorted flat (N, D) point cloud in to (B, max_num_pts, D) tensor.

    This function requires points to be ordered by batch.

    Args:
        pos: (N, 3)
        bi: ()

    """
    assert pos.shape[0] == bi.shape[0]
    assert pos.shape[-1] == 3, "For now only usage on 3D positions to find KNN neighbours is expected."
    N, D = pos.shape
    bi = bi.long()
    torch._assert_async((bi[1:] >= bi[:-1]).all())

    ptr = torch.searchsorted(bi, torch.arange(B + 1, device=pos.device, dtype=bi.dtype))
    starts, num_pts = ptr[:-1], ptr.diff()
    in_sample_idx = torch.arange(N, device=pos.device) - starts[bi]

    ptcl_padded = pos.new_zeros(B, int(num_pts.amax()), D)
    ptcl_padded[bi, in_sample_idx] = pos
    return _RaggedPtcl(ptcl_padded, num_pts, starts, in_sample_idx)


def knn_indices_ragged(q_pos, q_bi, c_pos, c_bi, K, batch_size=None):
    """Per-sample KNN using pytorch3d ragged batching.

    Returns:
        (Nq, K) indices of nearest neighbours for each probec point.
    """
    assert q_pos.shape[-1] == c_pos.shape[-1] == 3
    assert q_bi.shape[0] == q_pos.shape[0]
    assert c_bi.shape[0] == c_pos.shape[0]
    B = batch_size if batch_size is not None else int(c_bi.amax()) + 1

    q = _to_ragged(q_pos, q_bi, B)
    c = _to_ragged(c_pos, c_bi, B)

    # (B, max_nq, K), neighbour indices local to each sample's padded row.
    _, local_idx, _ = knn_points(
        q.ptcl_padded, c.ptcl_padded, lengths1=q.num_pts, lengths2=c.num_pts, norm=2, K=K, return_nn=False, return_sorted=True
    )

    # The clamp guards the degenerate case of a sample holding fewer than K context points.
    return (local_idx[q_bi, q.in_sample_idx] + c.starts[q_bi].unsqueeze(-1))


def _spread_on_sphere_low(n):
    """Hardcoded optimal point spreads for n <= 6.
    
    Spread over the x-y axis is preferred over z direction spread wherever possible because most
    interesting variations happen over x-y in driving data.
    """
    assert n <= 6
    s3 = math.sqrt(3)
    if n == 1:                                    # single point, in x-y plane
        out = [(1., 0., 0.)]
    if n == 2:                                    # antipodal pair in x-y plane   180°
        out = [(1., 0., 0.), (-1., 0., 0.)]
    if n == 3:                                    # equilateral triangle, x-y plane 120°
        out = [(1., 0., 0.), (-0.5, s3 / 2, 0.), (-0.5, -s3 / 2, 0.)]
    if n == 4:                                    # regular tetrahedron (isotropic) 109.47°
        c = 1. / s3
        out = [(c, c, c), (c, -c, -c), (-c, c, -c), (-c, -c, c)]
    if n == 5:                                    # triangular bipyramid            90°
        out = [(1., 0., 0.), (-1., 0., 0.),                     # apex pair, antipodal in x-y
                (0., 1., 0.), (0., -0.5, s3 / 2), (0., -0.5, -s3 / 2)]  # triangle in y-z plane
    if n == 6:                                    # regular octahedron (isotropic)  90°
        out =  [(1., 0., 0.), (-1., 0., 0.),
                (0., 1., 0.), (0., -1., 0.),
                (0., 0., 1.), (0., 0., -1.)]
    return torch.tensor(out, dtype=torch.float32)

def _dirty_fibonacci_sphere_spread(n):
    """Kind of maximally spread a number of points over a sphere.
    I looked up how to spread points over a sphere maximally and it needs fancy slow optimization loops,
    which is super boring if we are only going to use it for initialization.

    Instead of fancy slow optimization we dirtball it with the golden ratio and the Hat Box theorem:
    https://mathworld.wolfram.com/ArchimedesHat-BoxTheorem.html.

    Conceptually we want to have the maximum amount of surface area for each point on the sphere. We factorize
    it first by just linspacing points over the height (from pole to pole) of the sphere so we get one height per
    pole. Then we will also generate azimuth angles for every point.

    The factorization (first sampling height, then angles) works because of the linked Hat Box theorem. Given that
    the surface on the sphere between two vertical slices is always the same linspacing the height and then spreding
    points on every slice works. In approximation if slices are thin enough it should then be OK to linspace angles
    on each slice.

    Just taking a linspace of angles on each slice would lead to spokes like a bicycle wheel at the same positions but
    we want different angles for every height slice for a maximal spread. We can try to make a spiral, but we want a
    spiral that never ends up (or as slowly as possible) with the same angle after several spiral rotations. It turns
    out that the most chaotic number (or) most irrational is the golden ratio  (1 + sqrt(5))/2. If we rotate in units
    of the golden ratio (so 2pi * the ratio) then we will essentially never get the same angle again at each rotation.
    """
    i = torch.arange(n, dtype=torch.float32)
    z =  1 - 2 * (i + 0.5)/n # heights
    radii = torch.sqrt((1 - z*z).clamp(min=0))
    golden_ratio = (1 + 5**0.5)/2
    azimuths = 2 * math.pi * i / golden_ratio
    x = radii * azimuths.cos()
    y = radii * azimuths.sin()
    return torch.stack([x, y, z], dim=-1)

def sphere_spread(n):
    if n <= 6:
        return _spread_on_sphere_low(n)
    return _dirty_fibonacci_sphere_spread(n)



# TODO: We can make this multi-scale to use the full multi-scale context to
# make predictions.
# TODO: Also make an initial feature guesser with Knn + merge
class DeformKnnAttn(nn.Module):

    def __init__(
        self,
        dim,
        num_offsets,
        num_samples,
        num_heads,
        voxel_size,
        mask_size,
        ff_mult = 4,
        learnable_interpolation: Literal["dist_and_feature", "dist_only", "dist_and_feature_diff", "None"] = "dist_and_feature",
        qk_attn=False,
        qk_norm=False,
        sdpa_gate=False,
        gradient_checkpointing = False,
        condition_prediction = False,
        adaln_attn_alpha: bool = True,
        adaln_ffn_alpha: bool = True,
        adaln_alpha_plus_one: bool = False,
        conditioning: Literal["adaln", "film"] = "adaln",
    ):
        assert dim%num_heads == 0
        assert learnable_interpolation in ["dist_and_feature", "dist_only", "dist_and_feature_diff", "None"]
        assert not (qk_attn and learnable_interpolation != "None"), "Learnable interpolation is not supported with qk attention."
        assert conditioning in ("adaln", "film"), f"Unknown conditioning {conditioning}."
        super().__init__()

        use_adaln = condition_prediction and conditioning == "adaln"
        use_film = condition_prediction and conditioning == "film"
        self._adaln_attn_alpha = adaln_attn_alpha
        self._adaln_ffn_alpha = adaln_ffn_alpha
        self._adaln_alpha_plus_one = adaln_alpha_plus_one
        # Shift and scale for both branches, plus the alpha gate of each gated branch.
        self._num_mod = 4 + int(adaln_attn_alpha) + int(adaln_ffn_alpha)
        if use_adaln:
            self._adaln_zero_pred = nn.Sequential(nn.GELU(), nn.Linear(dim, self._num_mod*dim))
            nn.init.zeros_(self._adaln_zero_pred[-1].weight)
            nn.init.zeros_(self._adaln_zero_pred[-1].bias)
        else:
            self._adaln_zero_pred = None

        # AdaLN-zero replaces the affine of the norms it modulates. FiLM modulates the branch outputs instead, so the
        # norms keep their affine as in the runs from before AdaLN-zero.
        self._q_ln = nn.LayerNorm(dim, elementwise_affine=(not use_adaln))
        self._kv_ln = nn.LayerNorm(dim)
        # TODO we can merge this and _p_attn_weights into one.
        self._p_offsets = nn.Linear(dim, 3*num_offsets*num_heads)

        # Make sure we predict no context-dependent extra offsets initially.
        nn.init.zeros_(self._p_offsets.weight)

        # Start by concentrically probing around the context point in all directions
        # Each head has its own direction like in the original deformable detr paper.
        # (H, 3)
        sphere_points = sphere_spread(num_heads)

        # We multiply all offsets by the voxel size in the forward pass so we should not account
        # for it in the initialization. To make sure we probe outside of a typical mask we offset
        # by half the mask size so that a point in the middle of the masked region starts probing
        # context just outside of the masked region
        sphere_points *= (mask_size/2)

        # For each head the points are spaced progressively further.
        # (H, 3) -> (O, H, 3)
        sphere_points = sphere_points.unsqueeze(0) * torch.arange(1, num_offsets + 1).view(num_offsets, 1, 1)
        self._p_offsets.bias.data[:] = sphere_points.view(-1)

        self._qk_attn = qk_attn
        if qk_attn:
            self._pq = nn.Linear(dim, dim)
            self._pk = nn.Linear(dim, dim)
            self._p_rel_pos = nn.Linear(3, dim//num_heads)

            nn.init.xavier_normal_(self._pq.weight)
            nn.init.zeros_(self._pq.bias)
            nn.init.xavier_normal_(self._pk.weight)
            nn.init.zeros_(self._pk.bias)
        else:
            self._p_attn_weights = nn.Linear(dim, num_offsets*num_heads)

            # Make sure attention starts out uniform
            nn.init.zeros_(self._p_attn_weights.weight)
            nn.init.zeros_(self._p_attn_weights.bias)

        #self._p_interpolate = nn.Linear(2*dim + 3, 1)
        #self._p_interpolate = nn.Linear(dim + 3, 1)
        self._pv = nn.Linear(dim, dim)
        # Use the standard glorot initialization for the value and output projections
        nn.init.xavier_normal_(self._pv.weight)
        nn.init.zeros_(self._pv.bias)

        # One per head or shared over heads? This has to be defined before _po for DDP ordering.
        if learnable_interpolation in ["dist_and_feature", "dist_and_feature_diff"]:
            self._p_k_merge_w = nn.Linear(dim//num_heads + 4, 1)
        elif learnable_interpolation == "dist_only":
            self._p_k_merge_w = nn.Linear(4, 1)
        else:
            self._p_k_merge_w = None
        self._learnable_interpolation = learnable_interpolation

        self._po = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self._po.weight)
        nn.init.zeros_(self._po.bias)

        # Initialize the Knn weighting to be uniform
        if self._p_k_merge_w is not None:
            nn.init.zeros_(self._p_k_merge_w.weight)
            nn.init.zeros_(self._p_k_merge_w.bias)

        self._num_heads = num_heads
        self._num_samples = num_samples
        self._num_offsets = num_offsets
        self._voxel_size = voxel_size

        # FiLM of the attention and FFN outputs on the condition embedding, starting at alpha = 1 and beta = 0. The
        # condition embedding has the predictor stage dim. "film" in the names keeps them out of weight decay.
        self._attn_ms_film = FiLM(dim, dim, use_encoder_in_condition=False) if use_film else None

        # The attention FFN
        self._ln_ffn = nn.LayerNorm(dim, elementwise_affine=(not use_adaln))
        self._lin_ffn_1 = nn.Linear(dim, dim*ff_mult)
        self._lin_ffn_2 = nn.Linear(dim*ff_mult, dim)

        self._ffn_ms_film = FiLM(dim, dim, use_encoder_in_condition=False) if use_film else None

        self._gradient_checkpointing = gradient_checkpointing

        
    def _deform_attn(
        self, q_feat, q_pos, q_bi, c_feat, c_pos, c_bi, alpha_attn = None, beta_attn = None, gamma_attn = None,
        offset_voxel_size = None, film_embed = None,
    ):
        """Args:
        
            q_feat: (Q, C)
            q_pos: (Q, 3)
            q_bi: (Q)
            c_feat: (N, C)
            c_pos: (N, 3)
            c_bi: (N)
            offset_voxel_size: Float or (B,) per-sample voxel size that the offsets are in units of. Defaults to the
                               voxel size given at construction.
            film_embed: (B, C) condition embedding of the FiLM conditioning.
        """
        assert q_feat.shape[-1] == c_feat.shape[-1]
        assert q_feat.shape[0] == q_pos.shape[0] == q_bi.shape[0]
        assert c_feat.shape[0] == c_pos.shape[0] == c_bi.shape[0]
        assert len(q_feat.shape) == len(q_pos.shape) == 2
        assert len(c_feat.shape) == len(c_pos.shape) == 2

        Q, C = q_feat.shape
        O = self._num_offsets
        H = self._num_heads

        normed_q = self._q_ln(q_feat)
        if beta_attn is not None:
            assert gamma_attn is not None
            normed_q = gamma_attn * normed_q + beta_attn

        # (Q, 1, 3) + (Q, O*H, 3) -> (Q, O*H, 3)
        # Predict offets in number of voxels so that deeper stages naturally get bigger offsets 
        # If we go multi-scale, the voxel size used in offset prediction should still depend on the prediction
        # stage and not the context stage. To predict at a deep stage we query in the center of masks, and relevant
        # context points will only exist relatively far away.
        offsets = self._p_offsets(normed_q).view(Q, O*H, 3)
        offset_voxel_size = self._voxel_size if offset_voxel_size is None else offset_voxel_size
        # (Q, 1, 1) per-query or a scalar.
        offset_voxel_size = offset_voxel_size[q_bi].view(Q, 1, 1) if torch.is_tensor(offset_voxel_size) else offset_voxel_size
        offsets = offsets * offset_voxel_size
        sample_pos = q_pos.unsqueeze(1) + offsets
        
        # Compute K nearest neighbour indices for each sampling position
        # pos: (Q, O*H, 3) -> (Q*O*H, 3)
        # bi: (Q,) -> (Q*O*H)
        sample_pos = sample_pos.view(-1, 3)
        sample_bi = q_bi.repeat_interleave(O*H)
        K = self._num_samples

        # (Q*O*H, K)
        knn = knn_indices_ragged(sample_pos, sample_bi, c_pos, c_bi, K)

        # (Q*O*H, K) -> (Q*O*K, H)
        knn = knn.view(Q*O, H, K).permute(0, 2, 1).contiguous().view(Q*O*K, H)

        # values projection and head split
        N, _  = c_feat.shape
        normed_c = self._kv_ln(c_feat)
        v = self._pv(normed_c)
        v = v.view(N, H, C//H)

        # Now we need to combine (Q*O*K, H) indices with the (N, H, C//H) values to get a 
        # (Q*O, H, K, C//H tensor with features). We need to correctly index the heads since
        # each of them has different sampling points.
        head_idx = torch.arange(H, device=normed_c.device).view(1, H).expand(Q*O*K, H)
        v = v[knn, head_idx] 

        # Now we want to combine the neighbours for each sampled point into a single
        # feature. We either just max pool them or do a learnable interpolation.
        if not self._qk_attn:
            v = v.view(Q*O, K, H, C//H).permute(0, 2, 1, 3).contiguous()
            if self._p_k_merge_w is not None:
                # First get the position of each of the neighbors (N, 3) -> (Q*O, H, K, 3)
                neighbor_pos = c_pos.unsqueeze(1).expand(N, H, 3)[knn, head_idx]
                neighbor_pos = neighbor_pos.view(Q*O, K, H, 3).permute(0, 2, 1, 3).contiguous()
                # We want the pairwise distances between query points (Q*O, H, 3) <-> (Q*O, H, K, 3) -> (Q*O, H, K, 3)
                # TODO: What if we just maxpool over the K neighbors? Maxpool often works well in practice.
                neighbor_dists =  sample_pos.view(Q*O, H, 1, 3) - neighbor_pos
                dist_norms = torch.linalg.vector_norm(neighbor_dists, dim=-1, keepdims=True).clamp_min(1e-3)
                neighbor_dists = neighbor_dists/dist_norms
                neighbor_dists = torch.cat([neighbor_dists, dist_norms], dim=-1)
                if self._learnable_interpolation == "dist_and_feature_diff":
                    # v: (Q*O, H, K, C//H), q_feat: (Q, C) -> expand to (Q*O, H, 1, C//H)
                    v_diff = v - q_feat.view(Q, 1, H, C//H).expand(Q, O, H, C//H).reshape(Q*O, H, 1, C//H)
                    interp_input = torch.cat([v_diff, neighbor_dists], dim=-1)
                elif self._learnable_interpolation == "dist_and_feature":
                    interp_input = torch.cat([v, neighbor_dists], dim=-1)
                elif self._learnable_interpolation == "dist_only":
                    interp_input = neighbor_dists
                else:
                    raise ValueError(f"Unknown learnable_interpolation {self._learnable_interpolation}")
                
                comb_weights = self._p_k_merge_w(interp_input)
                comb_weights = F.softmax(comb_weights, dim=-2) # Softmax over K neighbors

                # (Q*O, H, K, C//H) -> (Q*O, H, C//H)
                v = (comb_weights * v).sum(dim=-2)
                # Alternatively to the above we could predict attention weights for each of the K neighbors,
                # but the queries won't know the distances.
            else:
                # (Q*O, H, K, C//H) -> (Q*O, H, C//H)
                v = v.amax(dim=-2)

            # Now we want attention weights for each attention head with weights over
            # the O offsets summing to 1.
            # (Q, O*H) -> (Q, O, H) -> softmax(-2) -> (Q, O, H)
            attn_weights = F.softmax(self._p_attn_weights(normed_q).view(Q, O, H), dim=-2)
            attn_update = v.view(Q, O, H, C//H)
            # (Q, O, H, C//H) * (Q, O, H, 1) -> sum(1) -> (Q, H, C//H)
            attn_update = (attn_update * attn_weights.unsqueeze(-1)).sum(1).view(Q, C)

        else:
            # (Q, C) -> (Q, H, 1, C//H)
            q = self._pq(normed_q)
            q = q.view(Q, H, 1, C//H)

            k = self._pk(normed_c)
            k = k.view(N, H, C//H)
            k = k[knn, head_idx] 

            neighbor_pos = c_pos.unsqueeze(1).expand(N, H, 3)[knn, head_idx]
            neighbor_pos = neighbor_pos.view(Q*O, K, H, 3)
            neighbor_dists =  sample_pos.view(Q*O, 1, H, 3) - neighbor_pos
            k = k + self._p_rel_pos(neighbor_dists).view(Q*O*K, H, C//H)
            
            # (Q*O, K, H, C//H) -> (Q, H, O, K, C//H) ->(Q, H, K*O, C//H)
            k = k.view(Q, O, K, H, C//H).permute(0, 3, 1, 2, 4).contiguous().view(Q, H, K*O, C//H)
            v = v.view(Q, O, K, H, C//H).permute(0, 3, 1, 2, 4).contiguous().view(Q, H, K*O, C//H)

            # (Q, H, 1, C//H)
            attn_update = F.scaled_dot_product_attention(q, k, v).view(Q, C)

        # TODO fix memory layout
        attn_update = self._po(attn_update)

        if alpha_attn is not None:
            attn_update = self._gate(attn_update, alpha_attn)
        if self._attn_ms_film is not None:
            attn_update = self._attn_ms_film(film_embed, attn_update, q_bi)

        return q_feat + attn_update

    def _gate(self, branch_out, alpha):
        """Scale a residual branch output by its AdaLN alpha gate, or by 1 + alpha with adaln_alpha_plus_one."""
        if self._adaln_alpha_plus_one:
            # As x + alpha * x so that the branch output keeps its dtype, like the encoder's voxel size modulation.
            return branch_out + branch_out * alpha
        return branch_out * alpha

    def _ffn_forward(self, q_feat, q_bi, alpha_ffn=None, beta_ffn=None, gamma_ffn=None, film_embed=None):
        normed_q = self._ln_ffn(q_feat)
        if gamma_ffn is not None:
            assert beta_ffn is not None
            normed_q = gamma_ffn * normed_q + beta_ffn
        ffn_out = self._lin_ffn_2(F.gelu(self._lin_ffn_1(normed_q)))
        if alpha_ffn is not None:
            ffn_out = self._gate(ffn_out, alpha_ffn)
        if self._ffn_ms_film is not None:
            ffn_out = self._ffn_ms_film(film_embed, ffn_out, q_bi)
        return q_feat + ffn_out


    def forward(self, q_feat, q_pos, q_bi, c_feat, c_pos, c_bi, cond_embed=None, offset_voxel_size=None):
        alpha_attn, beta_attn, gamma_attn, alpha_ffn, beta_ffn, gamma_ffn = (None,)*6
        film_embed = None
        if self._adaln_zero_pred is not None:
            assert cond_embed is not None
            Q, C = q_feat.shape
            mods = list(self._adaln_zero_pred(cond_embed)[q_bi].view(Q, self._num_mod, C).unbind(1))
            # Order: [alpha_attn], beta_attn, gamma_attn, [alpha_ffn], beta_ffn, gamma_ffn, the alphas only for gated branches.
            alpha_attn = mods.pop(0) if self._adaln_attn_alpha else None
            beta_attn, gamma_attn = mods.pop(0), mods.pop(0)
            alpha_ffn = mods.pop(0) if self._adaln_ffn_alpha else None
            beta_ffn, gamma_ffn = mods
            # fp32 so that small gammas are not rounded away by 1 + gamma in bfloat16.
            gamma_attn = 1 + gamma_attn.float()
            gamma_ffn = 1 + gamma_ffn.float()
        elif self._attn_ms_film is not None:
            assert cond_embed is not None
            film_embed = cond_embed
        if self._gradient_checkpointing:
            q_feat = checkpoint(
                self._deform_attn, q_feat, q_pos, q_bi, c_feat, c_pos, c_bi, alpha_attn, beta_attn, gamma_attn,
                offset_voxel_size, film_embed, use_reentrant=False, preserve_rng_state=True,
            )
            return checkpoint(
                self._ffn_forward, q_feat, q_bi, alpha_ffn, beta_ffn, gamma_ffn, film_embed,
                use_reentrant=False, preserve_rng_state=True,
            )
        return self._ffn_forward(
            self._deform_attn(
                q_feat, q_pos, q_bi, c_feat, c_pos, c_bi, alpha_attn, beta_attn, gamma_attn, offset_voxel_size, film_embed,
            ),
            q_bi,
            alpha_ffn,
            beta_ffn,
            gamma_ffn,
            film_embed,
        )
    

# Make max pool and no max pool version
class KnnInitializer(nn.Module):
    def __init__(self, k: int, dim: int, learnable_interpolation: bool = True):
        super().__init__()

        self._dim = dim
        self._pe = PEDynamic(embed_dim=dim, pos_dim=3)
        self._pos_proj = nn.Linear(3 + dim, dim)
        self._k = k
        if learnable_interpolation:
            self._merge_w_proj = nn.Linear(dim, 1)
        else:
            self._merge_w_proj = None
        self._context_proj = nn.Linear(dim, dim)

    def forward(self, q_pos, q_bi, c_feat, c_pos, c_bi, mask_size_embedding=None):

        assert q_pos.shape[0] == q_bi.shape[0]
        assert c_feat.shape[0] == c_pos.shape[0] == c_bi.shape[0]
        assert c_feat.shape[-1] == self._dim
        assert len(q_pos.shape) == len(c_feat.shape) == len(c_feat.shape) == 2
        assert len(q_bi.shape) == len(c_bi.shape) == 1
        # (Q, C)
        Q, _ = q_pos.shape
        _, C = c_feat.shape
        pos_feat = self._pos_proj(torch.cat([q_pos, self._pe(q_pos)], dim=-1))

        # (Q, K)
        knn = knn_indices_bruteforce(q_pos, q_bi, c_pos, c_bi, K=self._k)

        # (Q*K, C) -> (Q, K, C)
        context_update = c_feat[knn.view(-1)].view(Q, self._k, C)

        # (Q, K, C) -> (Q, C)
        if self._merge_w_proj is not None:
            w = F.softmax(self._merge_w_proj(context_update), dim=-2)
            context_update = (context_update * w).sum(-2)
        else:
            context_update = context_update.amax(-2)
        
        return self._context_proj(context_update) + pos_feat



        

class DeformPredictor(nn.Module):


    # TODO: Experiment with group size
    def __init__(
        self,
        encoder_stage_dims: list,
        predictor_stage_dims: list,
        voxel_size: float,
        mask_sizes: int,
        first_pred_stage: int = 0,
        predictor_head_dim: int = 32,
        layers_per_stage: int = 3,
        num_points: int = 3,
        num_neighbors: int = 4,
        learnable_interpolation: Literal["dist_and_feature", "dist_only", "dist_and_feature_diff", "None"] = "dist_and_feature",
        knn_init: bool = False,
        qk_attn: bool = False,
        condition_prediction: bool = False,
        condition_num_sweeps: bool = False,
        num_sweeps_embedding_table: bool = True,
        max_num_sweeps: int = 10,
        num_sweeps_embedding_init_std: float | None = 0.02,
        num_sweeps_query_input: bool = False,
        adaln_attn_alpha: bool = True,
        adaln_ffn_alpha: bool = True,
        adaln_alpha_plus_one: bool = False,
        conditioning: Literal["adaln", "film"] = "adaln",
        pred_before_unet: bool = False,
        condition_voxel_size: bool = False,
        voxel_size_query_input: bool = False,
        offset_scaling: Literal["sample", "reference", "none"] = "sample",
        reference_voxel_size: float | None = None,
        voxel_size_range: tuple[float, float] | None = None,
        voxel_size_embedding_init_std: float | None = 0.02,
        pred_deepest_mid: bool = False,
        target_stage_dims: list[int] | None = None,
    ):
        """
        Args:
            encoder_stage_dims: List of feature dimensions at each stage of the encoder.
            predictor_stage_dims: List of feature dimensions at each stage of the predictor.
                                In I-JEPA, V-JEPA, and V-JEPA 2 the predictor always has only 384
                                channels, regardless of the size of the encoder.
            voxel_size: Default (encoder input) voxel size, used whenever no voxel size is passed to forward.
            layers_per_stage: Number of attention layers per stage in the predictor.
            condition_prediction: Condition (see conditioning) on the mask size embeddings.
            condition_num_sweeps: Condition (see conditioning) on the number of aggregated sweeps.
            num_sweeps_embedding_table: Embed the number of sweeps with a table, otherwise a linear
                                layer on log2(num_sweeps).
            max_num_sweeps: Largest number of sweeps the embedding table has to support.
            num_sweeps_query_input: Append the number of sweeps as an extra input dimension to the queries.
            adaln_attn_alpha: Gate the attention output with the AdaLN-zero alpha. If False the attention
                                residual branch is ungated and only shift/scale its input.
            adaln_ffn_alpha: Gate the FFN output with the AdaLN-zero alpha, which starts at 0 so that the FFN
                                starts switched off. If False the FFN residual branch is ungated and only shift/scale
                                its input.
            adaln_alpha_plus_one: Apply the enabled AdaLN alpha gates as 1 + alpha, so that they start at 1 (open)
                                instead of 0.
            conditioning: How the condition embedding modulates every layer. "adaln": AdaLN-zero (shift/scale of the
                          non-affine layernorm outputs and alpha gates of the branch outputs). "film": FiLM of the
                          attention and FFN outputs (alpha starts at 1, beta at 0) with affine layernorms, as in the runs
                          from before AdaLN-zero.
            pred_before_unet: Whether to predict the features before and after the unet.
            condition_voxel_size: Condition (see conditioning) on the normalised log input voxel size, with a separate embedding per stage.
                                If several conditionings are on, their embeddings are summed and projected.
            voxel_size_query_input: Append the normalised log voxel size as an extra input dimension to the queries.
            offset_scaling: Units of the deformable offsets: each "sample"'s stage voxel size, the "reference"
                            stage voxel size, or "none" (meters, also for the initial probe distances).
            reference_voxel_size: Voxel size used by offset_scaling="reference". Only None if another offset_scaling is used.
            voxel_size_range: (min, max) training voxel sizes, mapped to u in [-1, 1] for the voxel size inputs. None
                              (no voxel size sampling) gives u = log2(voxel_size / default voxel size), i.e. 0 at the
                              default and +-1 per factor 2. Must match between training and evaluation.
            voxel_size_embedding_init_std: Init std of the voxel size embedding's output layer. None keeps the default init.
            pred_deepest_mid: With pred_before_unet, also predict the deepest stage's mid-stage features after its
                              output features.
            target_stage_dims: Feature dimensions of the targets at each stage. Defaults to encoder_stage_dims.
        """
        super().__init__()
        assert len(encoder_stage_dims) == len(predictor_stage_dims), (
            f"Expect the same number of encoder and corresponding predictor stages, "
            f"but got {len(encoder_stage_dims)} and {len(predictor_stage_dims)}."
        )
        assert all(d % 16 == 0 for d in predictor_stage_dims), (
            f"We divide attention heads per 16 channels but got predictor dimensionalities of {predictor_stage_dims}."
        )
        assert learnable_interpolation in ["dist_and_feature", "dist_only", "dist_and_feature_diff", "None"]

        self._stage_dims = predictor_stage_dims
        self._num_stages = len(predictor_stage_dims)
        self._condition_prediction = condition_prediction
        self._condition_num_sweeps = condition_num_sweeps
        self._num_sweeps_query_input = num_sweeps_query_input
        self._max_num_sweeps = max_num_sweeps
        assert not knn_init, "The KNN query initialisation was removed; queries start from a projection of their raw positions."
        assert offset_scaling in ("sample", "reference", "none")
        assert offset_scaling != "reference" or reference_voxel_size is not None
        self._offset_scaling = offset_scaling
        self._reference_voxel_size = reference_voxel_size

        if condition_num_sweeps:
            if num_sweeps_embedding_table:
                self._num_sweeps_embed = nn.ModuleList([nn.Embedding(max_num_sweeps + 1, pd) for pd in predictor_stage_dims])
                if num_sweeps_embedding_init_std is not None:
                    for e in self._num_sweeps_embed:
                        nn.init.normal_(e.weight, std=num_sweeps_embedding_init_std)
            else:
                self._num_sweeps_embed = nn.ModuleList([nn.Linear(1, pd) for pd in predictor_stage_dims])
        else:
            self._num_sweeps_embed = None
        self._num_sweeps_embedding_table = num_sweeps_embedding_table

        self._condition_voxel_size = condition_voxel_size
        self._voxel_size_query_input = voxel_size_query_input
        if condition_voxel_size:
            self._voxel_size_embed = nn.ModuleList(
                [VoxelSizeEmbedding(pd, out_init_std=voxel_size_embedding_init_std) for pd in predictor_stage_dims]
            )
        else:
            self._voxel_size_embed = None
        self._voxel_size_range = voxel_size_range

        if (int(condition_prediction) + int(condition_num_sweeps) + int(condition_voxel_size)) > 1:
            self._cond_embed_proj = nn.ModuleList([nn.Linear(pd, pd) for pd in predictor_stage_dims])
        else:
            self._cond_embed_proj = None

        # Define input projections before attention for more streamlined DDP communication structure
        # No layernorm is added on top of this in I-JEPA and V-JEPA. I suppose so that the whole
        # structure stays as close to prenorm as possible.
        self._context_projs = nn.ModuleList([nn.Linear(ed, pd, bias=True) for ed, pd in zip(encoder_stage_dims, predictor_stage_dims, strict=False)])

        # Queries start from a projection of their raw (metric) positions and any extra per-sample inputs.
        self._init_blocks = nn.ModuleList(
            [nn.Linear(3 + int(num_sweeps_query_input) + int(voxel_size_query_input), pd) for pd in predictor_stage_dims]
        )
        self._stage_voxel_sizes = [(voxel_size * 2 ** (i + 1 + first_pred_stage)) for i, _ in enumerate(predictor_stage_dims)]

        self._deform_attn_blocks = nn.ModuleList([])

        for i, (pd, ed, ms) in enumerate(zip(predictor_stage_dims, encoder_stage_dims, mask_sizes)):
            self._deform_attn_blocks.append(
                nn.ModuleList(
                    [
                        DeformKnnAttn(
                            dim=pd,
                            num_offsets=num_points,
                            num_samples=num_neighbors,
                            num_heads = pd // predictor_head_dim,
                            voxel_size=self._stage_voxel_sizes[i],
                            mask_size=ms,
                            ff_mult=4,
                            gradient_checkpointing=False,
                            learnable_interpolation=learnable_interpolation,
                            qk_attn=qk_attn,
                            condition_prediction=(condition_prediction or condition_num_sweeps or condition_voxel_size),
                            adaln_attn_alpha=adaln_attn_alpha,
                            adaln_ffn_alpha=adaln_ffn_alpha,
                            adaln_alpha_plus_one=adaln_alpha_plus_one,
                            conditioning=conditioning,
                        )
                        for _ in range(layers_per_stage)
                    ]
                )
            )
        
        self._voxel_size = voxel_size
        self._first_pred_stage = first_pred_stage

        # A learned bias can absorb this offset; revisit if adding Fourier encodings or other positional features.
        # self._mask_tokens = nn.ModuleList([nn.Parameter(torch.randn(sd)) for sd in stage_dims])

        target_stage_dims = encoder_stage_dims if target_stage_dims is None else target_stage_dims
        assert len(target_stage_dims) == len(encoder_stage_dims)
        self._out_projs = nn.ModuleList([])
        for i, (pd, ed) in enumerate(zip(predictor_stage_dims, target_stage_dims, strict=False)):
            # Note that if our predictor is narrower than the encoder (not the case yet) the bias
            # can not be absorbed by the affine norm.
            # The deepest stage has no decoder, so its second target (if any) is its mid-stage features.
            two_targets = i < len(encoder_stage_dims) - 1 or pred_deepest_mid
            mult = 2 if pred_before_unet and two_targets else 1
            self._out_projs.append(nn.Sequential(nn.LayerNorm(pd), nn.Linear(pd, mult * ed, bias=True)))

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
    def condition_prediction(self):
        return self._condition_prediction

    @property
    def requires_num_sweeps(self):
        return self._condition_num_sweeps or self._num_sweeps_query_input

    def _embed_num_sweeps(self, embed, num_sweeps):
        if self._num_sweeps_embedding_table:
            assert (num_sweeps >= 1).all()
            assert (num_sweeps <= self._max_num_sweeps).all()
            return embed(num_sweeps.long())
        return embed(num_sweeps.float().log2().view(-1, 1))

    @torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    def forward(
        self,
        student_feat_stages: list[torch.Tensor],
        student_pos_stages: list[torch.Tensor],
        masked_teacher_pos_stages: list[torch.Tensor],
        student_batch_indices_stages: list[torch.Tensor],
        masked_teacher_batch_indices_stages: list[torch.Tensor],
        register_tokens: list[torch.Tensor] | None = None,
        condition_embeddings: list[torch.Tensor] | None = None,
        num_sweeps: torch.Tensor | None = None,
        voxel_size: torch.Tensor | None = None,
    ):
        """Forward pass of the predictor.
        Args:
            student_feat_stages: List of student features at each stage. Each element is (N, C).
            student_pos_stages: List of student positions at each stage. Each element is (N, 3).
            masked_teacher_pos_stages: List of masked teacher positions at each stage. Each element is (M, 3)
            student_batch_indices_stages: List of batch indices for student points at each stage. Each element is (N,)
            masked_teacher_batch_indices_stages: List of batch indices for masked teacher points at each stage. Each element is (M,)
            condition_embeddings: List of (B, C) mask size embeddings at each stage.
            num_sweeps: (B,) number of aggregated sweeps in each sample.
            voxel_size: (B,) encoder input voxel size of each sample. The stage voxel sizes are derived from it.
                        Defaults to the voxel size given at construction.

            N and M are the concatinated points and masked points in the batch respectively.
        Returns:
            List of predicted teacher features at each stage. Each element is (M, C) where M is the number of
            masked points at that stage and C is the corresponding encoder stage dimension. With pred_before_unet
            all but the last stage are (M, 2C): the decoder output followed by the down-path features. With
            pred_deepest_mid the last stage is (M, 2C) too: its output followed by its mid-stage features.
        """
        assert len(student_feat_stages) == len(masked_teacher_pos_stages)
        assert len(student_feat_stages) == len(student_pos_stages)
        assert len(self._deform_attn_blocks) == len(student_pos_stages)
        if self.requires_num_sweeps:
            assert num_sweeps is not None, "The predictor is configured to use the number of sweeps but got None."
        if self._condition_voxel_size or self._voxel_size_query_input:
            assert voxel_size is not None, "The predictor is configured to use the voxel size but got None."
        outs = []
        num_stages = len(student_feat_stages)
        stage_factors = [2 ** (i + 1 + self._first_pred_stage) for i in range(num_stages)]
        # Float or (B,) per stage. Without a voxel size the construction default is used.
        sample_stage_voxel_sizes = [
            voxel_size.float() * f if voxel_size is not None else default for f, default in zip(stage_factors, self._stage_voxel_sizes, strict=True)
        ]
        if self._offset_scaling == "sample":
            offset_stage_voxel_sizes = sample_stage_voxel_sizes
        elif self._offset_scaling == "reference":
            offset_stage_voxel_sizes = [self._reference_voxel_size * f for f in stage_factors]
        else:
            offset_stage_voxel_sizes = [1.0] * num_stages

        # Do not feed the stage voxel size explicitly to the embedding module, as each stage already has its own embedding and can infer its stage.
        voxel_size_u = log_voxel_size_coordinate(voxel_size, self._voxel_size, self._voxel_size_range) if voxel_size is not None else None

        for (
            student_feat,
            student_pos,
            teacher_pos,
            student_bi,
            teacher_bi,
            deform_attn_blocks,
            context_proj,
            init_block,
            out_proj,
            mask_size_embed,
            num_sweeps_embed,
            cond_embed_proj,
            offset_stage_voxel_size,
            voxel_size_embed,
        ) in zip(
            student_feat_stages,
            student_pos_stages,
            masked_teacher_pos_stages,
            student_batch_indices_stages,
            masked_teacher_batch_indices_stages,
            self._deform_attn_blocks,
            self._context_projs,
            self._init_blocks,
            self._out_projs,
            condition_embeddings if self.condition_prediction else [None] * num_stages,
            self._num_sweeps_embed if self._num_sweeps_embed is not None else [None] * num_stages,
            self._cond_embed_proj if self._cond_embed_proj is not None else [None] * num_stages,
            offset_stage_voxel_sizes,
            self._voxel_size_embed if self._voxel_size_embed is not None else [None] * num_stages,
            strict=False,
        ):
            assert teacher_pos.shape[0] > 0, f"No masked points at this stage when encoder dim = {student_feat.shape[1]} Should not happen"

            num_sweeps_cond = self._embed_num_sweeps(num_sweeps_embed, num_sweeps) if num_sweeps_embed is not None else None
            if voxel_size_embed is not None:
                with torch.autocast("cuda", enabled=False):

                    # Each stage has its own embedding module, so it already "knows" its stage -> No need to feed it the stage voxel size explicitly.
                    # Feeding it the stage voxel size would only add a constant shift that its own MLP can absorb. Both inputs carry exactly the same information.
                    voxel_size_cond = voxel_size_embed(voxel_size_u)
            else:
                voxel_size_cond = None
            conds = [c for c in (mask_size_embed, num_sweeps_cond, voxel_size_cond) if c is not None]
            if len(conds) > 1:
                cond_embed = cond_embed_proj(sum(conds))
            elif len(conds) == 1:
                cond_embed = conds[0]
            else:
                cond_embed = None

            context_feats = context_proj(student_feat)

            q_in = [teacher_pos]
            if self._num_sweeps_query_input:
                q_in.append(num_sweeps[teacher_bi].to(teacher_pos.dtype).view(-1, 1))
            if self._voxel_size_query_input:
                q_in.append(voxel_size_u[teacher_bi].to(teacher_pos.dtype).view(-1, 1))
            q_feat = init_block(torch.cat(q_in, dim=-1))
            
            for _, deform_attn, in enumerate(deform_attn_blocks):
                q_feat = checkpoint(
                    deform_attn,
                    q_feat,
                    teacher_pos,
                    teacher_bi,
                    context_feats,
                    student_pos,
                    student_bi,
                    cond_embed=cond_embed,
                    offset_voxel_size=offset_stage_voxel_size,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
                out = q_feat
            outs.append(out_proj(out))

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
