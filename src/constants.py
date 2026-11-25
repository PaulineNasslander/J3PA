"""File for storing constants used throughout the project."""

import os

import numpy as np

MAX_LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 0.05 * MAX_LEARNING_RATE
EPOCHS = 100
EPOCHS_DEBUG = 10
BATCH_SIZE = 32 #Total BS should be 32*4=128 for same as prev runs (ema schedule optimized for). 
ACCUMULATED_STEPS = 1 # For pretraining.
DDP_BF16_GRAD_SYNC = False  # Opt in to BF16 gradient communication; parameters and optimizer states stay FP32.
# NOTE When adjusting num gpus or batch size per rank make sure to keep the effective batch size at 64 for probe training and 128 for pretraining.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 16))

PROBE_LEARNING_RATE = 5e-4
PROBE_BATCH_SIZE = 4
PROBE_HIDDEN_DIM = 512
PROBE_NUM_WORKERS = 8
PROBE_SEED = 42
PROBE_SCHEDULER_MAX_STEPS = 50000 #atm this is the largest of the eval at steps in offline eval might want to increase if we want to eval longer (?)

# The probe dataloaders are instantiated while the main dataloaders are still in memory. We avoid them
# from becoming too big.
EVAL_PROBE_NUM_WORKERS = int(os.environ.get("EVAL_PROBE_NUM_WORKERS", 8))
PREFETCH_FACTOR = 2
WARMUP_STEPS = 128000 # This is equal to 1k actual steps because the effective batch size is 128. Stupid legacy but now we use it:(

# Masking strategy
MASK_PROB = [0.15, 0.15, 0.3, 0.3, 0.3] # Masking probability for each stage. We can have different masking probabilities for different predictive stages.
MIN_MASK_SIZES = [4, 4, 4, 4, 4]# Number of voxels. Deeper stages have bigger voxels.
MAX_MASK_SIZES = [10, 10, 10, 10, 10] # Number of voxels.
ROTATE_MASK_SAMPLING = True
SHIFT_MASK_SAMPLING = True

# When masking we will mask given fractions of the content at each stage. How we define what content fraction
# is masked can be done in several ways:
# - point_cascade: The number of points inside each voxel at each stage is counted.
#                  Mask blocks are weighted by their point count until a given point fraction is masked.
# - fine_voxel_cascade: The number of points inside each voxel is counted by the number of points it had
#                       at the finest voxel stage. This is then propagated to mask blocks.
# - one_stage_hop: The number of points coming from the previous stage into the voxels at each stage is tracked. Also
#                   propagated to the mask blocks. For the finest stage we do not count points but each voxel counts as
#                   1. 
# - in_stage_count: We only count voxels within the masked blocks for each stage.
POINT_CASCADE = "point_cascade"
FINE_VOXEL_CASCADE = "fine_voxel_cascade"
ONE_STAGE_HOP = "one_stage_hop"
IN_STAGE_COUNT = "in_stage_count"
MASK_INFO_WEIGHTING_VALID = (POINT_CASCADE, FINE_VOXEL_CASCADE, ONE_STAGE_HOP, IN_STAGE_COUNT)
MASK_INFO_WEIGHTING = FINE_VOXEL_CASCADE # The actual option we choose.
WEIGHT_QUERY_COUNT_BY_MASKED_COUNT = True # If True we keep proportionally more queries for samples with more masked voxels at a stage.


# RoPE basis angle scaling
# TODO This is dead and not actually used
INITIAL_ALPHA = 1.0

# Teacher model
# The following 4 values have been experimented with in detail and the values below were found to be optimal.
EMA_START=0.9920
EMA_MAIN=0.9990
EMA_END=0.9995
EMA_WARMUP_STEPS=5000
assert EMA_START <= EMA_MAIN
assert EMA_END >= EMA_MAIN

VOXEL_SIZE = 0.05  # meters. Note that the initial pooling before the first attention layer is 2x2x2, so the effective voxel size is 5 cm for VOXEL_SIZE=2.5 cm.
# Voxel size as a parameter. VOXEL_SIZE above is always used in validation and the evaluations.
VOXEL_SIZE_RANGE = None # (min, max) in meters to sample a voxel size uniformly per training sample. None trains with VOXEL_SIZE only.
# The teacher always encodes at VOXEL_SIZE to avoid collapse mode only predicting voxelsize embedding.
FIXED_TEACHER_VOXEL_SIZE = True
# The voxel size inputs below use log2 of the voxel size, mapped linearly so that VOXEL_SIZE_RANGE spans [-1, 1].
CONDITION_ENCODER_ON_VOXEL_SIZE = False # Modulate the encoder attention layers.
VOXEL_SIZE_POINT_FEATURE = False # Append the voxel size as an extra feature to every input point of the encoder.
CONDITION_PRED_ON_VOXEL_SIZE = False # AdaLN-zero condition the predictor on the voxel size.
VOXEL_SIZE_QUERY_INPUT = False # Append the voxel size as an extra input dimension to the predictor queries.
VOXEL_SIZE_COND_INIT_STD = 0.02 # Init std of the layer producing the encoder modulations. 0 starts exactly unmodulated.
assert VOXEL_SIZE_RANGE is None or VOXEL_SIZE_RANGE[0] <= VOXEL_SIZE_RANGE[1]
# Whether to use fixed position normalization for the encoder or not. This affects the encoder input PE and RoPE (the
# predictor works in metric units).
# Grids, pooling and masks always use each sample's voxel size no matter the state of this flag. 
FIXED_POS_NORM_VOXEL_SIZE = True # True uses REFERENCE_VOXEL_SIZE for every sample, also in validation and evaluation for the position normalization stuff.
REFERENCE_VOXEL_SIZE = 0.05 # meters. Keep >= max(VOXEL_SIZE_RANGE)/pi so the top RoPE frequency does not alias.
# What unit to use for the predictor's deformable offsets: "sample"=voxel size of the current sample, "reference"=REFERENCE_VOXEL_SIZE, or "none"=meters / no scaling of offsets.
PRED_OFFSET_SCALING = "reference"
assert PRED_OFFSET_SCALING in ("sample", "reference", "none")

# Encoder versions 

# # 1B model
# ENCODER_STAGE_DIMS = [256, 512, 1024, 1536, 1664]
# ENCODER_NUM_DOWN_LAYERS = [6, 6, 6, 12, 12]
# ENCODER_NUM_UP_LAYERS = [2, 2, 2, 2]
# ENCODER_NUM_HEADS = [4, 8, 16, 24, 26]

# Ablations model
ENCODER_STAGE_DIMS = [64, 128, 256, 512, 1024]
ENCODER_NUM_DOWN_LAYERS = [3, 3, 3, 5, 4]
ENCODER_NUM_UP_LAYERS = [1, 1, 1, 1]
ENCODER_NUM_HEADS = [2, 4, 8, 16, 32]

# # Large
# ENCODER_STAGE_DIMS = [128, 192, 256, 512, 1024] 
# ENCODER_NUM_DOWN_LAYERS = [4, 4, 4, 10, 6]
# ENCODER_NUM_UP_LAYERS = [2, 2, 2, 2]
# ENCODER_NUM_HEADS = [2, 3, 4, 8, 16]

# # Small
# ENCODER_STAGE_DIMS = [64, 128, 256, 512, 704] 
# ENCODER_NUM_DOWN_LAYERS = [2, 2, 2, 2, 4]
# ENCODER_NUM_UP_LAYERS = [2, 2, 2, 2]
# ENCODER_NUM_HEADS = [1, 2, 4, 8, 11]

# # J3PA-Tiny
# ENCODER_STAGE_DIMS = [64, 128, 192, 256, 256]
# ENCODER_NUM_DOWN_LAYERS = [2, 2, 2, 2, 4]
# ENCODER_NUM_UP_LAYERS = [2, 2, 2, 2]
# ENCODER_NUM_HEADS = [2, 4, 6, 8, 8]

ENCODER_GROUP_SIZE = 1024
NUM_REGISTERS = 0
REGISTER_REDUCTION = "softmax"  # "amax", "sum" or "mean", or "softmax"
SAME_GRID = False # Whether to use the same grid in the teacher and student model or not.
USE_CONTEXT_WINDOW = True # Whether to use a context window or not. If true we will use a context window sampled based on min and max frac below else that code will not be run.
CONTEXT_MIN_FRAC = 0.8
CONTEXT_MAX_FRAC = 1.0 # For context window similar to I-JEPA set 1.0, for more local view style set lower. 
ROTATE_CONTEXT_WINDOW = True
SEP_REG_VOXEL_NORMS = False # Activate to experiment with specialized norm layers fr registers and voxel/point features.
USE_ATTENTION_BIAS = False # If true we will concat keys and queries with a learned bias vector.
ENC_QK_LAYERNORM = False
ENC_QK_LAYERNORM_IMPL = "triton_fused_rope" # "torch", "triton", or "triton_fused_rope"; execution only, including evaluation.
assert ENC_QK_LAYERNORM_IMPL in ("torch", "triton", "triton_fused_rope")
ENC_XSA = False # Exclusive self attention: remove the component of each token's attention output along its own value.
#NOTE turn of sdpa gate for the smaller models:
ENC_SDPA_GATE = True # Gate the attention output elementwise with a sigmoid of the token's input before the output projection.
ENC_POSTNORM_POOL = False # norm -> proj -> maxpool -> Gelu (prenorm) OR proj -> maxpool -> norm -> GeLU (postnorm)
ENC_RETURN_DOWN_PATH = True # Supervise pre and post Unet features.
ENC_QKV_BIAS = True
# Also supervise the deepest stage halfway through its layers, as a second target next to its output like the down-path
# targets of the decoded stages. Its student features are also passed to the predictor through the context combiner.
DEEPEST_MID_TARGET = True
assert not DEEPEST_MID_TARGET or ENC_RETURN_DOWN_PATH, "The mid target of the deepest stage is part of the down path."

# Predictor
# Conditioning of the predictor's layers on the mask size (and any other condition) embedding:
# "adaln" = AdaLN-zero: shift/scale of non-affine layernorm outputs and alpha gates (see ADALN_ATTN_ALPHA, ADALN_FFN_ALPHA,
#           ADALN_ALPHA_PLUS_ONE and ADALN_WEIGHT_DECAY).
# "film" = FiLM of the attention and FFN outputs (alpha starts at 1, beta at 0) with affine layernorms, as in the runs
#          from before AdaLN-zero. No weight decay, like all parameters with "film" in their name.
PRED_CONDITIONING = "film"
assert PRED_CONDITIONING in ("adaln", "film")

# Predictor versions:

# # 1B model
# PRED_STAGE_DIMS = [256, 512, 1024, 1536, 1664]
# PRED_HEAD_DIM = 64

# Ablations Model
PRED_STAGE_DIMS = [64, 128, 256, 512, 1024]
PRED_HEAD_DIM = 32

# # J3PA-Large
# PRED_STAGE_DIMS = [128, 192, 256, 512, 1024]
# PRED_HEAD_DIM = 64

# # J3PA-Small
# PRED_STAGE_DIMS = [64, 128, 256, 512, 704]
# PRED_HEAD_DIM = 64

# # J3PA-Tiny
# PRED_STAGE_DIMS = [64, 128, 192, 256, 256]
# PRED_HEAD_DIM = 32

assert len(PRED_STAGE_DIMS) == len(MASK_PROB), "We need to specify a masking probability for each predictor stage."
assert len(PRED_STAGE_DIMS) <= len(ENCODER_STAGE_DIMS), "We can have at most one predictor stage per encoder stage."

# WARNING: Less than 6 predictor layers seems to lead to model collapse around epoch 25-30.
# It is not clear why this happens at the moment. We have tested with 3 and 4 predictor layers
# and several (more aggressive) gradient norm clipping thresholds but got loss spikes followed
# by gradual collapse in all cases.
NUM_PRED_LAYERS = 3 # TODO if we have larger then 4 now we should implement re-permutation
EXP_NUM_SUPERVISION_QUERIES_PER_SAMPLE = 200
USE_REGISTERS_IN_PREDICTOR = False
# QUERY_INTERACTION=False
# PRED_QK_NORM=True
# PRED_SDPA_GATE=True
PRED_LEARNED_INTERPOLATION="dist_only"  # Options: "dist_and_feature", "dist_only", "dist_and_feature_diff", "None"
PRED_NUM_NEIGHBORS=21
PRED_NUM_OFFSETS=3
PRED_QK_ATTN=False
assert not (USE_REGISTERS_IN_PREDICTOR and NUM_REGISTERS == 0), "Registers can not be used in the predictor if there are no register tokens."

# LOSS
ALIGN_REGISTERS = False
REGISTER_LOSS_WEIGHT = 0.03
assert not (ALIGN_REGISTERS and NUM_REGISTERS == 0), (
    "A loss to align teacher and student registers can not be applied if there are no register tokens."
)
NORM_TEACHER = True # Nonaffine norm on teacher like in I-JEPA.
LOSS_DIM_SCALING = "sqrt" # Options are "lin", "sqrt", "equal". Sqrt option gives same gradient norm per stage.
LOSS_NUM_Q_WEIGHT = False # Weights the loss contribution by the actual number of queries that was kept

# Mask conditioning of the predictor.
MASK_EMBEDDING_TABLE = True # If False an MLP operating on the log2 of the mask sizes is used instead.
MASK_EMBEDDING_INIT_STD = 0.02 # None keeps the nn.Embedding default N(0, 1).
ADALN_WEIGHT_DECAY = False # Weight decay on the predictor AdaLN conditioning projections.
# This effectively sets encoder gradients to zero at intialization which seems dangerous.
ADALN_ATTN_ALPHA = False # If False the predictor attention output is not gated by the AdaLN-zero alpha.
# The alpha starts at 0 (see ADALN_ALPHA_PLUS_ONE), so with the gate the FFN branches start switched off and their
# weights get no gradient at first.
ADALN_FFN_ALPHA = True # If False the predictor FFN output is not gated by the AdaLN-zero alpha.
ADALN_ALPHA_PLUS_ONE = False # If True the enabled predictor alpha gates are applied as 1 + alpha, so they start at 1 instead of 0.
CONDITION_PRED_ON_MASK_SIZE = True # Whether to condition the predictor on the mask size embedding or not.

# Point cloud and augmentations.
USE_TIME = True
XYZ_START = int(USE_TIME)
XYZ_SLICE = slice(XYZ_START, XYZ_START + 3)
X_INDEX = XYZ_START
Y_INDEX = XYZ_START + 1
Z_INDEX = XYZ_START + 2
REFL_INDEX = XYZ_START + 3
POINT_FEATURE_DIM = 4 + int(USE_TIME) # 4 for x, y, z, intensity + 1 if using time
PC_RANGE = [-50, 50, -50, 50, -10, 10]
# Per dataset [(x_min, x_max), (y_min, y_max)] in meters that the pre-training crop centre is drawn from.
RANDOM_CROP_CENTER_RANGES = {
    "ZOD": [(-80, 80), (-80, 80)],
    "Argoverse2": [(-50, 50), (-50, 50)],
    "NuScenes": [(-20, 20), (-20, 20)],
    "TruckScenes": [(-100, 100), (-100, 100)],
    "OctoSense": [(-50, 50), (-50, 50)],
    "PandaSet": [(-40, 40), (-40, 40)],
    "KITTI": [(-20, 20), (-20, 20)],
    "Cirrus": [(0, 100), (-20, 20)],
}
MAX_NUM_POINTS = 300000
MIN_NUM_POINTS = 20000

# Augmentation values
# x, y, z rotation augmentation range in radians
FULL_ROTATION_AUG_RANGE = [
    [-np.pi / 64, np.pi / 64],  # x-axis rotation range
    # TODO They only rotate up to pi/64 in Unota to retain the gravity prior
    [-np.pi / 64, np.pi / 64],  # y-axis rotation range
    [-np.pi, np.pi],  # z-axis rotation range
]
MAX_TRANSLATION = 20 # meters
MAX_TIME_SHIFT = 0.0 # seconds

REFL_DROP_PROB = 0.0  # Per-sample drop probability of setting all intensities to 0 in the common augmentations. Value of 0 disables it.
REFL_NOISE_STD = 10  # Intensity is 0..255, so this is 4% of its range.

USE_SCALE_AUG = True
SCALE_AUG_RANGE = [0.95, 1.05]  # Scale augmentation range


# Per dataset (min, max) sweeps aggregated per pre-training sample, the count drawn uniformly from [min, max].
NUM_SWEEPS_RANGES = {
    "ZOD": (1, 1),
    "Argoverse2": (1, 2),
    "NuScenes": (1, 3),
    "TruckScenes": (1, 1),
    "OctoSense": (1, 2),
    "PandaSet": (1, 2),
    "KITTI": (1, 2),
    "Cirrus": (1, 1),  # No ego poses, so always a single scan.
}
assert all(1 <= low <= high for low, high in NUM_SWEEPS_RANGES.values())
NUM_EVAL_SWEEPS = 1  # Evaluation only uses the labelled key sweep.

CONDITION_PRED_ON_NUM_SWEEPS = False # Condition the predictor on the number of sweeps (summed with the mask size embedding and projected if both are on).
NUM_SWEEPS_EMBEDDING_TABLE = True # If False a linear layer on log2(num_sweeps) is used instead.
NUM_SWEEPS_QUERY_INPUT = False # Append the number of sweeps as an extra input dimension to the predictor queries.
MAX_NUM_SWEEPS_EMBED = max(max(high for _, high in NUM_SWEEPS_RANGES.values()), NUM_EVAL_SWEEPS)

# Paths:
# Per-point ZOD labels written by `scripts/precompute_zod_labels.py`. One
# `<frame_id>.npz` per frame, aligned with the raw lidar point order.
ZOD_LABELS_DIR = os.environ.get("ZOD_LABELS_DIR", "/datasets/labels/zod")
# Per-point Argoverse 2 labels written by `scripts/precompute_argoverse2_labels.py`.
# One `<split>/<log_id>/<timestamp_ns>.npz` per sweep, aligned with the raw sweep order.
ARGOVERSE2_LABELS_DIR = os.environ.get(
    "ARGOVERSE2_LABELS_DIR", "/datasets/labels/argoverse2"
)
# Per-point TruckScenes labels written by `scripts/precompute_truckscenes_labels.py`. One `.npz` per key-frame
# LiDAR file, at the .pcd's path relative to the dataset root (samples/<channel>/...), in raw .pcd point order.
TRUCKSCENES_LABELS_DIR = os.environ.get(
    "TRUCKSCENES_LABELS_DIR", "/datasets/labels/truckscenes"
)
# Per-point KITTI labels written by `scripts/precompute_kitti_labels.py`. One `<object frame>.npz` per labelled
# object benchmark frame, in raw scan order.
KITTI_LABELS_DIR = os.environ.get("KITTI_LABELS_DIR", "/datasets/labels/kitti")
# Per-point Cirrus labels written by `scripts/precompute_cirrus_labels.py`. One `dataset<k>/<timestamp_ns>.npz` per
# annotated gaussian scan, in raw scan order.
CIRRUS_LABELS_DIR = os.environ.get("CIRRUS_LABELS_DIR", "/datasets/labels/cirrus")

# Backend override used for the frozen teacher during distillation. Set to None to use the backend
# saved with the teacher checkpoint, falling back to "triton" for older runs.
DISTILL_TEACHER_QK_LAYERNORM_IMPL = "triton_fused_rope"
assert DISTILL_TEACHER_QK_LAYERNORM_IMPL in (None, "torch", "triton", "triton_fused_rope")

# Related to online evaluation:
YAW_ROTATION_AUG_RANGE = [-np.pi, np.pi]
MAX_NUM_BOXES = 500
TP_ERROR_NAMES = ["ATE", "ASE", "AOE"]
N_TP_ERRORS = len(TP_ERROR_NAMES)
N_REC_SAMPLES = 101
MAX_SCALE_ERROR: float = 1.0
MAX_YAW_ERROR: float = np.pi
TP_THRESH = 2.0  # distance used to compute TP statistics
MIN_RADIUS = 7.0  # used to be 2.0 incresed for denseOD
SCORE_THRESHOLD = 0.5
MAX_NUM_OBJECTS = 200
# Higher is better.
MIN_AP: float = 0.0
MIN_CDS: float = 0.0
# Lower is better.
MAX_NORMALIZED_ATE: float = 1.0
MAX_NORMALIZED_ASE: float = 1.0
MAX_NORMALIZED_AOE: float = 1.0
SUMMARY_DEFAULT_VALS = np.array([MIN_AP, TP_THRESH, MAX_NORMALIZED_ASE, MAX_NORMALIZED_AOE, MIN_CDS])
TP_NORMALIZTION_TERMS = np.array([TP_THRESH, MAX_SCALE_ERROR, MAX_YAW_ERROR])
SIGNIFICANT_DIGITS = 3
STATISTIC_NAMES = ["AP"] + TP_ERROR_NAMES + ["CDS"]
AFFINITY_THRESH = [0.5, 1.0, 2.0, 4.0]  # Meters
FSOD_VEHICLE_AP = 0.355
FSOD_VEHICLE_CDS = 0.675
