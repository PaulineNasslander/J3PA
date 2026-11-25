"""Object Detection Head."""

import torch
import torch.nn.functional as F
from torch import nn

from data.zod_dataset import CLASS_NAME_TO_ID as ZOD_CLASS_NAME_TO_ID
from j3pa.rope import RoPE
from utils.geometry import quaternion_to_yaw
POSITION_OFFSET_DIM = 3  # offset from point to box center x,y,z
SIZE_DIM = 3  # l,w,h
ORIENTATION_DIM = 2  # sin,cos of yaw angle
from torch.utils.checkpoint import checkpoint

class GlobalSelfAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion_factor: int | float,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = dim // num_heads

        self._ln_attn = nn.LayerNorm(dim)
        self._pqkv = nn.Linear(dim, 3 * dim, bias=False)
        self._po = nn.Linear(dim, dim)
        self._ln_ffn = nn.LayerNorm(dim)
        self._ffn = nn.Sequential(
            nn.Linear(dim, expansion_factor * dim, bias=(expansion_factor > 1)),
            nn.GELU(),
            nn.Linear(expansion_factor * dim, dim),
        )

    def forward(self, pt_feat, batch_indices, rope: RoPE, register=None):
        assert batch_indices.shape[0] == pt_feat.shape[0]
        assert pt_feat.shape[-1] == register.shape[-1] if register is not None else True, "Point features and register features must have the same dimension"
        B = batch_indices.max().item() + 1
        N, C = pt_feat.shape
        qkv = self._pqkv(self._ln_attn(pt_feat)).view(N, 3, C)
        q = rope.apply_precomputed(qkv[:, 0])
        k = rope.apply_precomputed(qkv[:, 1])
        v = qkv[:, 2]
        qkv_roped = torch.stack([q, k, v], dim=1)  # (N, 3, C)
        

        attn_out_full = torch.empty_like(pt_feat)
        if register is not None:
            B, R, C = register.shape

            # Why is this somehow messed up if we don't do this the pooling scatter complains.
            register = register.to(dtype=pt_feat.dtype)

            # (B, R, 3C) -> (B, R, 3, H, C//H) -> 3, (B, H, R, C//H)
            rq, rk, rv = self._pqkv(self._ln_attn(register)).view(B, R, 3, self._num_heads, C // self._num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        else: 
            R = 0

        for b in range(B):
            sample_mask = (batch_indices == b)
            N_b = sample_mask.sum().item()

            # (N, 3, C) -> 3 x (1, H, N_b, C//H) for multi-head attention
            q, k, v = (
                qkv_roped[sample_mask]
                .view(1, N_b, 3, self._num_heads, C // self._num_heads) 
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )
        
            if register is not None:
                q = torch.cat([q, rq[b:b+1]], dim=-2) # (1, H, N_b + R, C//H)
                k = torch.cat([k, rk[b:b+1]], dim=-2) # (1, H, N_b + R, C//H)
                v = torch.cat([v, rv[b:b+1]], dim=-2) # (1, H, N_b + R, C//H)

            # (1, H, N_b + R, C//H) -> (1, N_b + R, H, C//H) -> (1, N_b + R, C)
            attn_out = (
                nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
                .transpose(1, 2)
                .contiguous()
                .view(1, N_b + R, C)
            )
            # (1, N_b + R, C) -> (1, N_b, C), (1, R, C)
            # (1, N_b, C) -> (N_b, C)
            attn_out_pt = attn_out[:, :N_b, :].contiguous().squeeze(0)
            
            # We do not need to update the registers since we only do onle layer and they are not used in the last od layer.
            attn_out_full[sample_mask] = attn_out_pt

        pt_feat = pt_feat + self._po(attn_out_full)
        pt_feat = pt_feat + self._ffn(self._ln_ffn(pt_feat))
        return pt_feat
    

class ODHead(nn.Module):
    """Small detection head for online evaluation.

    Predicts for each point:
    - objectness score (1)
    - position offset to box center (3)
    - size (3)
    - orientation as sin and cos of yaw angle (2)
    Total output dim: 9
    Does not include classification.
    """

    def __init__(self, in_feat_dim, hidden_dim=512, attention=False, use_registers=False, num_classes=None):
        """Initialize."""
        super().__init__()
        self._box_out_dims = POSITION_OFFSET_DIM + SIZE_DIM + ORIENTATION_DIM  # = 8
        self._obj_out_dims = num_classes if num_classes is not None else len(ZOD_CLASS_NAME_TO_ID)  # number of classes
        self._layer1 = nn.Linear(in_feat_dim, hidden_dim)

        self._attention = attention
        if attention:
            self._attention_layer = GlobalSelfAttn(
                            dim=hidden_dim,
                            num_heads=hidden_dim // 32,
                            expansion_factor=4,
                        )
            self._rope = RoPE(dim=hidden_dim, pos_dim=3, learnable_scale=True)
            self._reg_proj = nn.Linear(in_feat_dim-3, hidden_dim) if use_registers else None # minus 3 since registers has no position
        else:
            self._activation = nn.ReLU()

        self._box_layer = nn.Linear(hidden_dim, self._box_out_dims)
        self._obj_layer = nn.Linear(hidden_dim, self._obj_out_dims)

    def forward(self, point_feats: list[torch.Tensor], up_indices: list[torch.Tensor], 
                pos_at_s0: torch.Tensor, batch_idx_at_s0: torch.Tensor, 
                registers: list[torch.Tensor] | None = None
                ) -> torch.Tensor:
        """Forward pass.

        Args:
            point_feats: list of (M, in_feat_dim) tensors of point features, where M is total points across batches
            up_indices: list of (M, 1) or (M,) tensors of indices to upcast features to original point cloud
            pos_at_s0: (N, 3) tensor of positions at stage 0
            batch_idx_at_s0: (N,) tensor of batch indices for each point in the point cloud at stage 0
            registers: (B, R, register_dim) tensor of register tokens if using registers in od head.
        
        Returns:
            (N, obj_out_dims + box_out_dims) tensor of predictions where N is the number of points in the original point cloud
        """
        point_feats = [f.to(torch.bfloat16) for f in point_feats]
        x = self.parameterless_upcast(point_feats, up_indices, pos_at_s0)
       
        batch_indices = batch_idx_at_s0
        
        # Ensure batch_indices is 1D
        if batch_indices.dim() > 1:
            batch_indices = batch_indices.squeeze()
        
        if registers is not None:
            registers = [r.to(torch.bfloat16) for r in registers]
            registers = torch.cat(registers, dim=-1) # list[(B, R, register_dim)] -> (B, R, total_register_dim)
            B, R, C = registers.shape
            if self._attention:
                registers = self._reg_proj(registers.view(-1, C)).view(B, R, -1) # (B, R, hidden_dim)
            else:
                #TODO some fix to add registers to the point features
                pass
        
        # Forward pass for this batch
        feats = self._layer1(x) #first layer in bfloat16 for memory saving when using the concat feature, rest in float32 for stability
        
        if self._attention:
            self._rope.cache_cos_sin(feats, pos_at_s0)
            feats = checkpoint(
                self._attention_layer,
                feats, batch_indices, rope=self._rope, register=registers,
                use_reentrant=False, 
                preserve_rng_state=True,
            )  
        else:
            feats = self._activation(feats)
        
        with torch.autocast("cuda", enabled=False):  # ensure head is in float32 for stability
            feats = feats.float()  # Ensure features are in float32 for the head
            output = torch.cat((self._obj_layer(feats), self._box_layer(feats)), dim=-1)
            
        return output
    
    def forward_no_upcast(self, point_feats: torch.Tensor, batch_idx_at_s0: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            point_feats: (N, in_feat_dim) tensor of point features, where N is total points across batches
            batch_idx_at_s0: (N,) tensor of batch indices for each point in the point cloud at stage 0
            
        Returns:
            (N, obj_out_dims + box_out_dims) tensor of predictions where N is the number of points in the original point cloud
        """
        point_feats = point_feats.to(torch.bfloat16)
        batch_indices = batch_idx_at_s0
        
        # Ensure batch_indices is 1D
        if batch_indices.dim() > 1:
            batch_indices = batch_indices.squeeze()
        
        # Forward pass for this batch
        feats = self._layer1(point_feats) #first layer in bfloat16 for memory saving when using the concat feature, rest in float32 for stability
        feats = self._activation(feats)
        
        with torch.autocast("cuda", enabled=False):  # ensure head is in float32 for stability
            feats = feats.float()  # Ensure features are in float32 for the head
            output = torch.cat((self._obj_layer(feats), self._box_layer(feats)), dim=-1)
            
        return output

    @staticmethod
    def parameterless_upcast(features: list[torch.Tensor], up_indices: list[torch.Tensor], pos_at_s0: torch.Tensor) -> torch.Tensor:
        """Utility function to upcast encoded features without aggregation nor learnable parameters."""
        # upcast_feat = features[-1][up_indices[-1]]  # start with the deepest features upsampled to the previous stage
        # for feat, up_idx in zip(reversed(features[:-1]), reversed(up_indices[:-1])):  # noqa: B905
        #     upcast_feat = torch.cat((upcast_feat, feat), dim=-1)
        #     upcast_feat = upcast_feat[up_idx]
            
        # upcast_feat = torch.cat((upcast_feat, org_point_cloud[..., 1:4]), dim=-1)
        # return upcast_feat
        n_points = pos_at_s0.shape[0]
        total_dim = sum(f.shape[-1] for f in features) + 3  # +3 for xyz

        out = torch.empty(n_points, total_dim, dtype=features[0].dtype, device=features[0].device)

        composed_indices = [None]
        for i in range(1, len(up_indices)):
            composed = up_indices[i]
            for j in range(i-1):
                composed = composed[up_indices[i-j-1]]
            composed_indices.append(composed)
        # Write each feature slice directly into the output buffer
        col = 0
        for feat, idx in zip(reversed(features), reversed(composed_indices), strict=True):            
            d = feat.shape[-1]
            if idx is None:
                out[:, col:col+d] = feat
            else:
                assert idx.min() >= 0 and idx.max() < feat.shape[0], "Index out of bounds in upcast"
                out[:, col:col+d] = feat[idx]
            col += d
        out[:, col:] = pos_at_s0
        return out
    
    
    def loss(self, pos, preds, annotations, batch_indices):
        """Compute loss."""
        centers = annotations["center"]  # (B, M, 3)
        sizes = torch.relu(annotations["size"]) + 1e-7  # (B, M, 3)
        quats = annotations["quats"]  # (B, M, 4)
        classes = annotations["class"]  # (B, M)

        # go from position offset to absolute position in preds
        abs_preds = preds.clone()
        abs_preds[:, self._obj_out_dims : self._obj_out_dims + POSITION_OFFSET_DIM] += pos  # (N, 3)
        abs_preds = abs_preds.to(torch.float32)

        # split predictions
        obj_p = abs_preds[:, : self._obj_out_dims]  # (N, obj_out_dims)
        box_p = abs_preds[:, self._obj_out_dims :]  # (N, box_out_dims)

        obj_loss = torch.tensor(0.0, device=obj_p.device, dtype=torch.float32)
        box_loss = torch.tensor(0.0, device=obj_p.device, dtype=torch.float32)

        for b_idx in range(centers.shape[0]):
            # Process each batch separately
            unpad_mask = (centers[b_idx] != 0).any(dim=-1)  # (M,)
            centers_b = centers[b_idx][unpad_mask]  # (M', 3)
            sizes_b = sizes[b_idx][unpad_mask]  # (M', 3)
            quats_b = quats[b_idx][unpad_mask]  # (M', 4)
            classes_b = classes[b_idx][unpad_mask]  # (M',)

            batch_mask = batch_indices == b_idx
            obj_p_b = obj_p[batch_mask]  # (N_b, obj_out_dims)
            box_p_b = box_p[batch_mask]  # (N_b, box_out_dims)
            pos_b = pos[batch_mask]  # (N_b, 3)

            if centers_b.shape[0] == 0:
                #print("No objects in this frame.")
                obj_t_b = torch.zeros_like(obj_p_b, device=obj_p_b.device, dtype=torch.float32)
                obj_loss += F.binary_cross_entropy_with_logits(
                    obj_p_b,
                    obj_t_b,
                    reduction="mean",
                )
                continue

            # get targets:
            orientation_b = quaternion_to_yaw(quats_b).unsqueeze(-1)  # (M', 1)
            box_t_b = torch.cat([centers_b, sizes_b, torch.sin(orientation_b), torch.cos(orientation_b)], dim=-1)  # (M', 8)
            radii = torch.max(box_t_b[:, 3:6], dim=-1)[0] / 2.0  # (M',)

            # Initialize targets
            obj_t_b = torch.zeros_like(obj_p_b)  # (N_b, num_classes)
            closest_target_idx = torch.full((pos_b.shape[0],), -1, dtype=torch.long, device=pos_b.device)
            min_dists = torch.full((pos_b.shape[0],), float('inf'), device=pos_b.device)

            # Process each target individually to avoid large distance matrix
            for i, (center, radius, cls) in enumerate(zip(centers_b, radii, classes_b, strict=True)):
                dists = torch.norm(pos_b - center.unsqueeze(0), dim=-1)  # (N_b,)
                in_radius = dists <= radius  # (N_b,)
                
                # Assign class
                obj_t_b[in_radius, cls - 1] = 1.0
                
                # Track closest target for box regression
                closer = dists < min_dists
                update_mask = in_radius & closer
                closest_target_idx[update_mask] = i
                min_dists[update_mask] = dists[update_mask]

            # === Object Classification Loss ===
            pos_weight = torch.ones(obj_p_b.shape[-1], device=obj_p_b.device) * 2.0
            obj_loss += F.binary_cross_entropy_with_logits(
                obj_p_b,
                obj_t_b,
                pos_weight=pos_weight,
                reduction="mean",
            )

            # === Box Regression Loss ===
            fg_mask = closest_target_idx >= 0
            if fg_mask.any():
                box_p_fg = box_p_b[fg_mask]  # (K, 8)
                box_t_fg = box_t_b[closest_target_idx[fg_mask]]  # (K, 8)
                box_loss += F.l1_loss(box_p_fg, box_t_fg, reduction="mean")
            else:
                print(f"No foreground points for box regression. Although there are GTs in batch {b_idx}.")

        return obj_loss, box_loss
