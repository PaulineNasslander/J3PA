import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.multi_gpu_utils import (
    get_gpu_device,
    is_main_process,
    torch_ddp_active,
)
from utils.utils import filter_and_flatten, get_dataloader
from constants import (
    XYZ_SLICE,
    PROBE_BATCH_SIZE,
    PROBE_HIDDEN_DIM,
    PROBE_LEARNING_RATE,
    PROBE_NUM_WORKERS,
    PROBE_SEED,
    PROBE_SCHEDULER_MAX_STEPS,
)
from j3pa.pt_transf import PointTransformer
from utils.multi_gpu_utils import (
    get_gpu_device,
    is_main_process,
)
from evaluation.probe_utils import rotate_cloud, limit_pc_to_fov, _calc_mavg_loss, _cleanup_dataloader
from j3pa.pt_transf import PointTransformer
from torch.utils.tensorboard import SummaryWriter
from evaluation.heads import SemanticSegmentationHead
from evaluation.upcast import upcast_to_input_resolution

def pointwise_cls_eval(
    encoder: PointTransformer,
    version,
    epoch,
    num_train_steps=5000,
    learning_rate=PROBE_LEARNING_RATE,
    writer: SummaryWriter = None,
    dataset_type = "ZOD",
    nl_probe = False
):
    num_train_steps = num_train_steps if version == "full" else min(num_train_steps, 10)
    dim = sum(encoder._stage_dims) + 3

    cls_train_loader, _ = get_dataloader(
        train_flag=True,
        version=version,
        batch_size=PROBE_BATCH_SIZE,
        num_workers=PROBE_NUM_WORKERS,
        prefetch_factor=1,
        shuffle=True, # Shuffle even tho seed bc we could see more then one pass over the data here for small dataset and then we don't want them in the same order every pass
        drop_last=True,
        load_annotations=True,
        dataset_type=dataset_type,
        load_segmentation=True,
        seed=PROBE_SEED,
    )
    cls_val_loader, _ = get_dataloader(
        train_flag=False,
        version=version,
        batch_size=PROBE_BATCH_SIZE,
        num_workers=PROBE_NUM_WORKERS,
        prefetch_factor=1,
        shuffle=False,
        drop_last=False,
        load_annotations=True,
        dataset_type=dataset_type,
        load_segmentation=True,
        seed=PROBE_SEED,
    )
    
    seg_class_name_to_id = cls_train_loader.dataset.get_seg_class_name_to_id()
    seg_ignore_id = cls_train_loader.dataset.get_seg_ignore_id()
    num_classes = len(seg_class_name_to_id) + 1
    head = SemanticSegmentationHead(
        in_feat_dim=dim,
        num_classes=num_classes,
        ignore_id=seg_ignore_id,
        hidden_dim=PROBE_HIDDEN_DIM if nl_probe else None,
    )
    head = head.to(device=get_gpu_device())
    head = DDP(head) if torch_ddp_active() else head

    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    sceduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, PROBE_SCHEDULER_MAX_STEPS, eta_min=learning_rate * 1e-3
    )

    # Ensure the encoder is in eval mode to avoid random permutations
    encoder_was_train = encoder.training
    encoder = encoder.eval()

    cls_train_iter = iter(cls_train_loader)
    prev_cls_loss = None
    for cls_step in range(num_train_steps):
        try:
            (point_cloud, annotations, field_of_view, seg_labels) = next(cls_train_iter)
        except StopIteration:
            cls_train_iter = iter(cls_train_loader)
            (point_cloud, annotations, field_of_view, seg_labels) = next(cls_train_iter)
        print(f"CLS Training step {cls_step + 1}/{num_train_steps}", end="\r")
        
        point_cloud = point_cloud.to(device=get_gpu_device())  # (1,N,C)
        annotations = {k: v.to(device=get_gpu_device()) for k, v in annotations.items()}
        field_of_view = field_of_view.to(device=get_gpu_device())  # (1,2)

        point_cloud = limit_pc_to_fov(point_cloud, field_of_view) #sets points outside fov to zero
        point_cloud, annotations, _ = rotate_cloud(point_cloud, annotations)  # apply rot aug (safe as rot of zero pad points is still zero)
        padding_mask = torch.any(point_cloud != 0, dim=-1)
        point_cloud, batch_indices = filter_and_flatten(point_cloud)  # (N, C)
        semantic = seg_labels["semantic"].to(device=get_gpu_device())[padding_mask]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                feats, _, _, up_indices, _ = encoder(point_cloud, batch_indices)
            positions = point_cloud[..., XYZ_SLICE]
            input_feats = upcast_to_input_resolution(feats, up_indices, positions)
            cls_preds = head(input_feats)

        if torch_ddp_active():
            obj_loss = head.module.loss(cls_preds, semantic)
        else:
            obj_loss = head.loss(cls_preds, semantic)

        obj_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sceduler.step()

        mav_cls_loss = _calc_mavg_loss(prev_cls_loss, obj_loss.item())
        prev_cls_loss = mav_cls_loss

    if writer is not None:
        writer.add_scalar("CLS Loss/train", mav_cls_loss, epoch)

    # Now evaluate CLS head
    head.eval()
    device = get_gpu_device()
    tp_counts  = torch.zeros(num_classes, dtype=torch.long, device=device)
    fp_counts  = torch.zeros(num_classes, dtype=torch.long, device=device)
    fn_counts  = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct_fg = torch.zeros(1, dtype=torch.long, device=device)
    total_fg   = torch.zeros(1, dtype=torch.long, device=device)
    for eval_idx, (point_cloud, annotations, field_of_view, seg_labels) in enumerate(cls_val_loader):
        print(f"CLS Eval step {eval_idx + 1}/{len(cls_val_loader)}", end="\r")
        point_cloud = point_cloud.to(device=device)
        annotations = {k: v.to(device=device) for k, v in annotations.items()}
        field_of_view = field_of_view.to(device=device)
        
        point_cloud = limit_pc_to_fov(point_cloud, field_of_view) #sets points outside fov to zero
        padding_mask = torch.any(point_cloud != 0, dim=-1)
        point_cloud, org_batch_indices = filter_and_flatten(point_cloud)  # (N, C)
        semantic = seg_labels["semantic"].to(device=device)[padding_mask]

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                feats, pos_stages, _, up_indices, _ = encoder(point_cloud, org_batch_indices)
                positions = point_cloud[..., XYZ_SLICE]
                input_feats = upcast_to_input_resolution(feats, up_indices, positions)
                cls_preds = head(input_feats)

        labelled = torch.ones_like(semantic, dtype=torch.bool) if seg_ignore_id is None else semantic != seg_ignore_id
        accumulate_seg_counts(
            tp_counts, fp_counts, fn_counts, correct_fg, total_fg,
            cls_preds[labelled].float(), semantic[labelled],
        )

    # Reduce per-class counters across GPUs (cheap: num_classes scalars each)
    if torch_ddp_active():
        dist.all_reduce(tp_counts,  op=dist.ReduceOp.SUM)
        dist.all_reduce(fp_counts,  op=dist.ReduceOp.SUM)
        dist.all_reduce(fn_counts,  op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_fg, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_fg,   op=dist.ReduceOp.SUM)

    if is_main_process():
        seg_metrics = compute_seg_metrics_from_counts(
            tp_counts, fp_counts, fn_counts, correct_fg, total_fg,
            num_classes, seg_class_name_to_id,
        )
        miou = seg_metrics["mIoU"]
        macc = seg_metrics["mAcc"]
        all_acc = seg_metrics["allAcc"]

        print("CLS Segmentation evaluation results:")
        print(f"  mIoU  : {miou:.4f}")
        print(f"  mAcc  : {macc:.4f}")
        print(f"  allAcc: {all_acc:.4f}")
        print("  Per-class IoU:", {k: f"{v:.4f}" for k, v in seg_metrics["per_class_iou"].items()})
        print("  Per-class Acc:", {k: f"{v:.4f}" for k, v in seg_metrics["per_class_acc"].items()})

        if writer is not None:
            writer.add_scalar("CLS Seg/mIoU", miou, epoch)
            writer.add_scalar("CLS Seg/mAcc", macc, epoch)
            writer.add_scalar("CLS Seg/allAcc", all_acc, epoch)
            for cls_name, iou_val in seg_metrics["per_class_iou"].items():
                writer.add_scalar(f"CLS Seg/IoU_{cls_name}", iou_val, epoch)
            
    else:
        miou, macc, all_acc = None, None, None

    _cleanup_dataloader(cls_train_loader)
    _cleanup_dataloader(cls_val_loader)

    if encoder_was_train:
        encoder.train()

    return miou, macc, all_acc


def accumulate_seg_counts(tp_counts, fp_counts, fn_counts, correct_fg, total_fg, preds_logits, targets):
    """Update running TP/FP/FN counters from a single batch (in-place).

    Args:
        tp_counts, fp_counts, fn_counts: (num_classes,) long tensors on the same device as preds_logits
        correct_fg, total_fg: scalar long tensors tracking foreground accuracy numerator/denominator
        preds_logits: (N, num_classes) raw logit tensor
        targets: (N,) integer target tensor (0=background, 1..C=foreground)
    """
    pred_classes = preds_logits.argmax(dim=1)  # (N,)
    num_classes = tp_counts.shape[0]
    for c in range(num_classes):
        pred_c = pred_classes == c
        target_c = targets == c
        tp_counts[c] += (pred_c & target_c).sum()
        fp_counts[c] += (pred_c & ~target_c).sum()
        fn_counts[c] += (~pred_c & target_c).sum()
    fg_mask = targets > 0
    if fg_mask.any():
        correct_fg += (pred_classes[fg_mask] == targets[fg_mask]).sum()
        total_fg += fg_mask.sum()


def compute_seg_metrics_from_counts(tp_counts, fp_counts, fn_counts, correct_fg, total_fg, num_classes, class_name_to_id, include_background=False):
    """Compute mIoU, mAcc, and allAcc from pre-accumulated TP/FP/FN counters.

    Args:
        tp_counts, fp_counts, fn_counts: (num_classes,) long tensors
        correct_fg, total_fg: scalar long tensors for foreground accuracy
        num_classes: total number of classes including background
        class_name_to_id: dictionary mapping class names to their IDs
        include_background: if False (default), class 0 is excluded from mIoU/mAcc.

    Returns:
        dict with keys mIoU, mAcc, allAcc, per_class_iou, per_class_acc
    """
    id_to_name = {v: k for k, v in class_name_to_id.items()}  # 1-indexed foreground
    id_to_name[0] = "background"

    per_class_iou = {}
    per_class_acc = {}
    iou_vals = []
    acc_vals = []

    start_class = 0 if include_background else 1
    for c in range(start_class, num_classes):
        tp = tp_counts[c].item()
        fp = fp_counts[c].item()
        fn = fn_counts[c].item()

        if (tp + fn) == 0:
            continue  # class absent from GT — skip

        iou = tp / (tp + fp + fn + 1e-8)
        acc = tp / (tp + fn + 1e-8)

        name = id_to_name.get(c, f"class_{c}")
        per_class_iou[name] = iou
        per_class_acc[name] = acc
        iou_vals.append(iou)
        acc_vals.append(acc)

    total_fg_val = total_fg.item()
    all_acc = (correct_fg.item() / total_fg_val) if total_fg_val > 0 else 0.0

    miou = float(torch.tensor(iou_vals).mean()) if iou_vals else 0.0
    macc = float(torch.tensor(acc_vals).mean()) if acc_vals else 0.0

    return {
        "mIoU": miou,
        "mAcc": macc,
        "allAcc": all_acc,
        "per_class_iou": per_class_iou,
        "per_class_acc": per_class_acc,
    }