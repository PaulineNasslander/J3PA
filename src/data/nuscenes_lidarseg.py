"""Label spaces for nuScenes-lidarseg and nuScenes-panoptic.

The raw releases label every point with one of 32 fine-grained categories. The
official benchmarks merge those into 16 evaluation classes plus an ``ignore``
class, and that merged space is what both the lidarseg and the panoptic
leaderboards report. Everything here converts between the two.

The raw category order follows nuScenes-lidarseg ``category.json``; the
evaluation mapping below implements this project's benchmark convention.

Convention in this repo: class id 0 means "ignore" and is excluded from metrics;
evaluation classes are 1..16. That matches the ``class_name_to_id`` convention
used by the object-detection heads, so the shared metric helpers work unchanged.
"""

import numpy as np

# The 32 fine-grained categories, in the index order used by the label files.
# This is the order of ``category.json`` in a lidarseg release.
LIDARSEG_RAW_CLASSES = (
    "noise",
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
    "static_object.bicycle_rack",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.motorcycle",
    "vehicle.trailer",
    "vehicle.truck",
    "flat.driveable_surface",
    "flat.other",
    "flat.sidewalk",
    "flat.terrain",
    "static.manmade",
    "static.other",
    "static.vegetation",
    "vehicle.ego",
)

# The 16 evaluation classes, in benchmark order. Ids are 1..16; 0 is ignore.
SEG_CLASSES = (
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
)
SEG_CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(SEG_CLASSES)}
NUM_SEG_CLASSES = len(SEG_CLASSES)
IGNORE_ID = 0

# "Thing" classes get instance ids in the panoptic labels; "stuff" classes do
# not. The split is fixed by the benchmark: the 10 detection classes are things.
THING_CLASSES = SEG_CLASSES[:10]
STUFF_CLASSES = SEG_CLASSES[10:]
THING_IDS = tuple(SEG_CLASS_NAME_TO_ID[name] for name in THING_CLASSES)
STUFF_IDS = tuple(SEG_CLASS_NAME_TO_ID[name] for name in STUFF_CLASSES)

# Fine-grained category -> evaluation class. Categories that the benchmark drops
# (rare, ambiguous or ego-vehicle points) map to ignore and are simply absent.
_RAW_NAME_TO_SEG_NAME = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
    "flat.driveable_surface": "driveable_surface",
    "flat.other": "other_flat",
    "flat.sidewalk": "sidewalk",
    "flat.terrain": "terrain",
    "static.manmade": "manmade",
    "static.vegetation": "vegetation",
}

# Lookup table indexed by raw category index, giving the evaluation class id.
RAW_INDEX_TO_SEG_ID = np.array(
    [
        SEG_CLASS_NAME_TO_ID.get(_RAW_NAME_TO_SEG_NAME.get(name, ""), IGNORE_ID)
        for name in LIDARSEG_RAW_CLASSES
    ],
    dtype=np.int64,
)


def raw_lidarseg_to_seg_ids(raw_labels: np.ndarray) -> np.ndarray:
    """Map raw 32-class lidarseg indices to the 16 evaluation class ids.

    Args:
        raw_labels: (N,) integer array of raw category indices.

    Returns:
        (N,) int64 array with 0 = ignore and 1..16 = evaluation classes.
    """
    labels = np.asarray(raw_labels, dtype=np.int64)
    # Indices beyond the known table can only come from a release we do not
    # understand; treat them as ignore rather than silently mislabelling them.
    out_of_range = labels >= RAW_INDEX_TO_SEG_ID.shape[0]
    labels = np.where(out_of_range, 0, labels)
    return RAW_INDEX_TO_SEG_ID[labels]


def split_panoptic_labels(panoptic_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split packed nuScenes-panoptic labels into semantic and instance ids.

    The labels pack ``category_index * 1000 + instance_index``. Which category
    indexing is used varies: the released files use the 16-class evaluation
    indexing, but labels generated from the raw categories use the 32-class one.
    Both are accepted, distinguished by whether any index exceeds 16.

    Args:
        panoptic_labels: (N,) integer array of packed labels.

    Returns:
        ``(semantic_ids, instance_ids)``, both (N,) int64. ``semantic_ids`` uses
        the evaluation class ids (0 = ignore). ``instance_ids`` is 0 for points
        without an instance.
    """
    packed = np.asarray(panoptic_labels, dtype=np.int64)
    category = packed // 1000
    instance = packed % 1000
    if category.size and category.max() > NUM_SEG_CLASSES:
        category = raw_lidarseg_to_seg_ids(category)
    # Instance ids are only meaningful where a class survives the merge.
    instance = np.where(category == IGNORE_ID, 0, instance)
    return category, instance
