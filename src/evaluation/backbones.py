"""Wrapper feature extractors /backbones for the models we want to benchmark."""

import torch
from torch import nn

from constants import ENC_QK_LAYERNORM_IMPL, USE_TIME, XYZ_SLICE
from evaluation.constants_loader import get_constant, load_constants_file
from j3pa.pt_transf import PointTransformer, PointSelfAttn

# training protocols:
FROZEN = "frozen"
FULL = "full"
TRAINABLE_MODES = (FROZEN, FULL)


class Backbone(nn.Module):
    """Common interface: raw points in, per-input-point features out."""

    def forward(self, point_cloud: torch.Tensor, batch_indices: torch.Tensor):
        """Encode a flattened batch of point clouds.

        Args:
            point_cloud: (N, C), [x, y, z, intensity] with optional leading time.
            batch_indices: (N,) sample index of each point.

        Returns:
            ``(features, up_indices, positions)`` where ``features`` is (N, D) aligned with
            the input points, ``up_indices`` are the upsampling indices, and ``positions`` is their (N, 3) xyz.
        """
        raise NotImplementedError

    @property
    def stage_dims(self) -> list[int]:
        """Channel dim of the feature at each encoder stage, shallowest first."""
        raise NotImplementedError
    
    def set_trainable(self, mode: str) -> None:
        """Freeze/unfreeze according to one of :data:`TRAINABLE_MODES`."""
        assert mode in TRAINABLE_MODES, f"Unknown trainable mode {mode!r}"
        self.requires_grad_(mode == FULL)
        
        # A frozen backbone must also stop updating norm statistics and stop
        # re-drawing the random space-filling-curve permutations.
        self.train(mode != FROZEN)


def _remap_legacy_rope_keys(state_dict: dict) -> dict:
    """Move hoisted RoPE weights onto the attention layers that now own them.

    Checkpoints written before "Optimize RoPE activation memory" kept one RoPE
    per attention layer in top-level ``_ds_ropes``/``_us_ropes`` lists and passed
    them into attention functionally; the modules are now submodules of the
    attention layers. The indices line up one-to-one and the `RoPE` state itself
    is unchanged, so only the key paths need rewriting.
    """
    if not any(key.startswith(("_ds_ropes.", "_us_ropes.")) for key in state_dict):
        return state_dict

    print("Checkpoint uses the pre-e7d49da RoPE layout, remapping its keys.")
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("_ds_ropes."):
            stage, layer, rest = key.removeprefix("_ds_ropes.").split(".", 2)
            key = f"_ds_attn_groups.{stage}.{layer}._rope.{rest}"
        elif key.startswith("_us_ropes."):
            layer, rest = key.removeprefix("_us_ropes.").split(".", 1)
            key = f"_us_attn_layers.{layer}._rope.{rest}"
        remapped[key] = value
    return remapped


# Every experiment rebuilds the backbone, so without this the same file is read
# once per (task, protocol) pair. A pre-training job that rotates its checkpoints
# can then delete it midway through an evaluation that has already run for hours.
_ENCODER_WEIGHTS_CACHE: dict[tuple[str, str], tuple[dict, int | None]] = {}


def _load_encoder_weights(checkpoint_path: str, weights_key: str) -> tuple[dict, int | None]:
    """Encoder weights plus pre-training epoch, read from disk once per process."""
    cache_key = (checkpoint_path, weights_key)
    if cache_key not in _ENCODER_WEIGHTS_CACHE:
        # Memory-mapped so only the encoder is read; the 1B runs' checkpoints are 22 GB, mostly optimizer state.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        if weights_key not in checkpoint:
            raise KeyError(
                f"{checkpoint_path} has no '{weights_key}' entry. Found: {sorted(checkpoint)}"
            )
        # Only the encoder is kept; the optimizer/predictor entries dominate the
        # file size and nothing here needs them.
        _ENCODER_WEIGHTS_CACHE[cache_key] = (
            _remap_legacy_rope_keys(checkpoint[weights_key]),
            checkpoint.get("epoch"),
        )
    return _ENCODER_WEIGHTS_CACHE[cache_key]


class J3PABackbone(Backbone):
    """The repo's own `PointTransformer`, restored from a training checkpoint.

    The architecture is rebuilt from the run's saved ``constants.py`` (or legacy ``constants_values.txt``) rather
    than from the current `src/constants.py`, so checkpoints stay loadable after
    the defaults move on. The QK-LayerNorm execution backend follows the current
    setting instead, so older checkpoints also use fusion by default.
    """

    def __init__(
        self,
        checkpoint_path: str,
        constants_path: str,
        use_teacher: bool = True,
        voxel_size: float | None = None,
    ):
        super().__init__()
        constants = load_constants_file(constants_path)
        if "INIT_PE_DIM_ENCODER" in constants:
            raise ValueError("Legacy YOLOFormer checkpoints (INIT_PE_DIM_ENCODER) are no longer supported.")
        checkpoint_use_time = constants.get("USE_TIME", True)
        if checkpoint_use_time != USE_TIME:
            raise ValueError(
                f"Checkpoint USE_TIME={checkpoint_use_time} does not match "
                f"src/constants.py USE_TIME={USE_TIME}. Match the setting before evaluation."
            )

        stage_dims = list(get_constant(constants, "ENCODER_STAGE_DIMS"))
        pred_stage_dims = list(get_constant(constants, "PRED_STAGE_DIMS"))
        # Mirrors train_jepa.py: the encoder only normalises (and upsamples to)
        # the stages the predictor actually supervises.
        out_norm_stages = list(
            reversed([len(stage_dims) - 1 - i for i in range(len(pred_stage_dims))])
        )
        # Runs from before the down/up split of the layers only recorded the total number of layers per stage.
        if "ENCODER_NUM_LAYERS" in constants and "ENCODER_NUM_DOWN_LAYERS" not in constants:
            depth = {"num_layers_per_stage": list(constants["ENCODER_NUM_LAYERS"])}
        else:
            depth = {
                "num_down_layers_per_stage": list(get_constant(constants, "ENCODER_NUM_DOWN_LAYERS")),
                "num_up_layers_per_stage": list(get_constant(constants, "ENCODER_NUM_UP_LAYERS")),
            }

        self.encoder = PointTransformer(
            stage_dims=stage_dims,
            **depth,
            num_heads_per_stage=list(get_constant(constants, "ENCODER_NUM_HEADS")),
            group_size=get_constant(constants, "ENCODER_GROUP_SIZE"),
            downsample_only=False,
            voxel_size=get_constant(constants, "VOXEL_SIZE"),
            num_registers=get_constant(constants, "NUM_REGISTERS"),
            register_reduction=get_constant(constants, "REGISTER_REDUCTION"),
            out_norm_stages=out_norm_stages,
            sep_reg_voxel_norms=get_constant(constants, "SEP_REG_VOXEL_NORMS"),
            use_attention_bias=get_constant(constants, "USE_ATTENTION_BIAS"),
            qk_layernorm=get_constant(constants, "ENC_QK_LAYERNORM"),
            qk_layernorm_impl=ENC_QK_LAYERNORM_IMPL,
            # Runs predating these options were trained without them.
            condition_voxel_size=constants.get("CONDITION_ENCODER_ON_VOXEL_SIZE", False),
            voxel_size_point_feature=constants.get("VOXEL_SIZE_POINT_FEATURE", False),
            pos_norm_voxel_size=constants.get("REFERENCE_VOXEL_SIZE") if constants.get("FIXED_POS_NORM_VOXEL_SIZE", False) else None,
            voxel_size_range=constants.get("VOXEL_SIZE_RANGE"),
            # Runs from before these options existed had neither, whatever the current defaults are. XSA has no
            # weights, so building it wrong would not be caught by loading the checkpoint.
            xsa=constants.get("ENC_XSA", False),
            pool_postnorm=constants.get("ENC_POSTNORM_POOL", False),
            qkv_bias=constants.get("ENC_QKV_BIAS", False),
            sdpa_gate=constants.get("ENC_SDPA_GATE", False),
        )
        self.voxel_size = get_constant(constants, "VOXEL_SIZE")
        # None keeps the encoder's own float default (and its historical rounding).
        self._eval_voxel_size = voxel_size
        if voxel_size is not None:
            self.voxel_size = voxel_size
        self.group_size = get_constant(constants, "ENCODER_GROUP_SIZE")
        # `PointTransformer` does not keep these around, so hold onto them here.
        self.num_heads_per_stage = list(get_constant(constants, "ENCODER_NUM_HEADS"))

        weights_key = "teacher_encoder" if use_teacher else "student_encoder"
        state_dict, self.pretrain_epoch = _load_encoder_weights(checkpoint_path, weights_key)
        # The down-path norms only feed the pretraining loss, so the encoder is built without them.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_down_layernorms.")}
        if state_dict["_init_proj.0.weight"].shape != self.encoder._init_proj[0].weight.shape:
            raise ValueError(
                f"{checkpoint_path} was trained with encoder input features that no longer exist (ENC_INPUT_VERSION="
                f"'new', or a run from before ENC_INPUT_VERSION existed). Only the raw point features + PE inputs remain."
            )
        self.encoder.load_state_dict(state_dict)
        print(
            f"Loaded {weights_key} from {checkpoint_path} "
            f"(pretraining epoch {self.pretrain_epoch})"
        )
        attention_layers = [layer for layer in self.encoder.modules() if isinstance(layer, PointSelfAttn)]
        fused_layers = sum(
            layer._q_ln is not None and layer._rope is not None and layer._qk_layernorm_impl == "triton_fused_rope"
            for layer in attention_layers
        )
        print(
            f"Evaluation QK-LayerNorm: enabled={get_constant(constants, 'ENC_QK_LAYERNORM')}, "
            f"backend={ENC_QK_LAYERNORM_IMPL}, "
            f"fused_rope_layers={fused_layers}/{len(attention_layers)}"
        )
    
    @property
    def feature_dim(self) -> int:
        """Width of the concatenated multi-stage feature, plus xyz."""
        return sum(self.encoder._stage_dims) + 3

    def forward(self, point_cloud, batch_indices, grid_idx=None):
        voxel_size_kwargs = {}
        if self._eval_voxel_size is not None:
            num_samples = int(batch_indices.max()) + 1
            voxel_size_kwargs["voxel_size"] = torch.full(
                (num_samples,), self._eval_voxel_size, device=point_cloud.device, dtype=torch.float32
            )
        stage_features, stage_positions, batch_indices, up_indices, _ = self.encoder(
            point_cloud, batch_indices, grid_idx=grid_idx, **voxel_size_kwargs
        )
        positions = point_cloud[..., XYZ_SLICE]
        return stage_features, stage_positions, batch_indices, up_indices, positions


def build_backbone(args) -> Backbone:
    """Instantiate the J3PA backbone from the evaluation arguments."""
    return J3PABackbone(
        checkpoint_path=args.checkpoint,
        constants_path=args.constants,
        use_teacher=not args.use_student,
        voxel_size=getattr(args, "voxel_size", None),
    )
