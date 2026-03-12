import argparse
import copy
import csv
import importlib
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mmcv
import numpy as np
import torch
from matplotlib import cm
from PIL import Image, ImageDraw
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize BEV segmentation (GT / prob / binary / TP-FP-FN)."
    )
    parser.add_argument("config", help="Config file path")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to visualize (default: train)",
    )
    parser.add_argument(
        "--show-dir",
        default=None,
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples to visualize",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated dataset indices (e.g. 1,5,10). If set, only these are visualized.",
    )
    parser.add_argument(
        "--seg-thresh",
        type=float,
        default=0.5,
        help="Threshold for binary segmentation",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Seg channel index for multi-channel seg. Use -1 for max across channels.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--force-train-mode",
        action="store_true",
        help="Force dataset test_mode=False to ensure gt_seg_mask availability.",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save raw prob/pred/gt arrays as npz",
    )
    return parser.parse_args()


def import_plugin(cfg: Config, config_path: str) -> None:
    if not hasattr(cfg, "plugin") or not cfg.plugin:
        return

    if hasattr(cfg, "plugin_dir"):
        module_dir = os.path.dirname(cfg.plugin_dir).split("/")
    else:
        module_dir = os.path.dirname(config_path).split("/")
    module_path = module_dir[0]
    for item in module_dir[1:]:
        module_path = module_path + "." + item
    importlib.import_module(module_path)


def maybe_unwrap_data_container(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 0:
            return value
        first = value[0]
        if hasattr(first, "data"):
            data = first.data
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            return data
        return first
    if hasattr(value, "data"):
        data = value.data
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data
    return value


def stack_batch_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) > 0:
        if all(torch.is_tensor(v) for v in value):
            return torch.stack(list(value), dim=0)
    return None


def select_current_metas(img_metas: Sequence[Any], len_queue: int) -> List[Dict[str, Any]]:
    current = []
    for item in img_metas:
        if isinstance(item, (list, tuple)):
            current.append(item[len_queue - 1])
        else:
            current.append(item)
    return current


def extract_seg_logits(
    model_module: torch.nn.Module,
    img: torch.Tensor,
    img_metas: Sequence[Any],
    points: Optional[Any],
) -> Tuple[Optional[torch.Tensor], List[Dict[str, Any]]]:
    if img.dim() == 5:
        img = img.unsqueeze(1)
    img = img.cuda(non_blocking=True)
    len_queue = img.size(1)

    lidar_feat = None
    if getattr(model_module, "modality", "vision") == "fusion" and points is not None:
        if isinstance(points, (list, tuple)):
            points = [p.cuda(non_blocking=True) for p in points]
        else:
            points = points.cuda(non_blocking=True)
        lidar_feat = model_module.extract_lidar_feat(points)

    prev_bev = None
    if len_queue > 1:
        prev_img = img[:, :-1, ...]
        prev_metas = copy.deepcopy(img_metas)
        prev_bev = model_module.obtain_history_bev(prev_img, prev_metas)

    cur_img = img[:, -1, ...]
    cur_metas = select_current_metas(img_metas, len_queue)
    img_feats = model_module.extract_feat(img=cur_img, img_metas=cur_metas)
    outs = model_module.pts_bbox_head(
        img_feats, lidar_feat, cur_metas, prev_bev=prev_bev
    )
    return outs.get("seg", None), cur_metas


def prepare_prob_and_gt(
    seg_logits: torch.Tensor,
    gt_seg: torch.Tensor,
    channel: int,
    thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # seg_logits: [B, C, H, W]
    # gt_seg: [B, C|1, H, W] or [B, H, W]
    if gt_seg.dim() == 3:
        gt_seg = gt_seg.unsqueeze(1)
    if gt_seg.dim() == 2:
        gt_seg = gt_seg.unsqueeze(0).unsqueeze(0)

    if gt_seg.shape[-2:] != seg_logits.shape[-2:]:
        gt_seg = torch.nn.functional.interpolate(
            gt_seg.float(),
            size=seg_logits.shape[-2:],
            mode="nearest",
        )

    pred_prob_all = torch.sigmoid(seg_logits.float())
    bsz, num_channels = pred_prob_all.shape[0], pred_prob_all.shape[1]

    if num_channels == 1:
        pred_prob = pred_prob_all[:, 0]
    else:
        if channel >= 0:
            channel = min(channel, num_channels - 1)
            pred_prob = pred_prob_all[:, channel]
        else:
            pred_prob = pred_prob_all.max(dim=1).values

    gt_channels = gt_seg.shape[1]
    if gt_channels == 1:
        gt_bin = gt_seg[:, 0] > 0.5
    else:
        if channel >= 0:
            gt_channel = min(channel, gt_channels - 1)
            gt_bin = gt_seg[:, gt_channel] > 0.5
        else:
            gt_bin = gt_seg.max(dim=1).values > 0.5

    pred_bin = pred_prob >= thresh
    return (
        pred_prob.detach().cpu().numpy(),
        pred_bin.detach().cpu().numpy().astype(np.bool_),
        gt_bin.detach().cpu().numpy().astype(np.bool_),
    )


def calc_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        iou=iou,
        f1=f1,
        gt_ratio=float(gt.mean()),
        pred_ratio=float(pred.mean()),
    )


def to_heatmap(prob: np.ndarray) -> np.ndarray:
    # prob: [H, W], values in [0, 1]
    return (cm.get_cmap("turbo")(prob)[..., :3] * 255.0).astype(np.uint8)


def draw_title(img: np.ndarray, title: str) -> np.ndarray:
    pil_img = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle((0, 0, pil_img.width, 28), fill=(0, 0, 0))
    draw.text((8, 7), title, fill=(255, 255, 255))
    return np.array(pil_img)


def to_mask_rgb(mask: np.ndarray) -> np.ndarray:
    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    img[mask] = (255, 255, 255)
    return img


def make_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # TP: green, FP: red, FN: blue
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, np.logical_not(gt))
    fn = np.logical_and(np.logical_not(pred), gt)
    vis = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    vis[tp] = (0, 255, 0)
    vis[fp] = (0, 0, 255)
    vis[fn] = (255, 0, 0)
    return vis


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.bool_)
    if m.shape[0] < 3 or m.shape[1] < 3:
        return m
    eroded = m.copy()
    eroded[1:-1, 1:-1] = (
        m[1:-1, 1:-1]
        & m[:-2, 1:-1]
        & m[2:, 1:-1]
        & m[1:-1, :-2]
        & m[1:-1, 2:]
    )
    return m & np.logical_not(eroded)


def draw_contours(base_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    vis = base_rgb.copy()
    gt_bd = mask_boundary(gt)
    pred_bd = mask_boundary(pred)
    # GT contour: blue, Pred contour: red
    vis[gt_bd] = (0, 0, 255)
    vis[pred_bd] = (255, 0, 0)
    return vis


def make_panel(
    prob: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    metrics: Dict[str, float],
    sample_name: str,
    thresh: float,
) -> np.ndarray:
    gt_vis = draw_title(to_mask_rgb(gt), "GT mask")
    prob_vis = draw_title(to_heatmap(prob), "Pred prob")
    pred_vis = draw_title(to_mask_rgb(pred), f"Pred mask @ {thresh:.2f}")
    err_vis = draw_title(make_error_map(pred, gt), "Error map (TP/FP/FN)")
    contour_vis = draw_title(draw_contours(to_heatmap(prob), gt, pred), "Contours (GT blue / Pred red)")

    text = np.zeros_like(gt_vis)
    text = draw_title(text, "Metrics")
    rows = [
        f"sample: {sample_name[:52]}",
        f"IoU: {metrics['iou']:.4f}",
        f"F1:  {metrics['f1']:.4f}",
        f"P/R: {metrics['precision']:.4f} / {metrics['recall']:.4f}",
        f"TP/FP/FN: {metrics['tp']} / {metrics['fp']} / {metrics['fn']}",
        f"GT ratio: {metrics['gt_ratio']:.4f}",
        f"Pred ratio: {metrics['pred_ratio']:.4f}",
    ]
    text_pil = Image.fromarray(text)
    draw = ImageDraw.Draw(text_pil)
    y = 40
    for line in rows:
        draw.text((8, y), line, fill=(230, 230, 230))
        y += 26
    text = np.array(text_pil)

    top = np.concatenate([gt_vis, prob_vis, pred_vis], axis=1)
    bottom = np.concatenate([err_vis, contour_vis, text], axis=1)
    return np.concatenate([top, bottom], axis=0)


def parse_indices(indices_text: str) -> Optional[set]:
    text = indices_text.strip()
    if not text:
        return None
    result = set()
    for part in text.split(","):
        part = part.strip()
        if part:
            result.add(int(part))
    return result


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg, args.config)

    set_random_seed(args.seed, deterministic=True)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.show_dir is None:
        args.show_dir = osp.join(
            "./work_dirs",
            osp.splitext(osp.basename(args.config))[0],
            "vis_bev_seg",
        )
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))

    logger = get_root_logger()
    logger.info("Saving BEV seg visualizations to %s", args.show_dir)

    dataset_cfg = copy.deepcopy(cfg.data[args.split])
    if isinstance(dataset_cfg, dict):
        if args.force_train_mode:
            dataset_cfg.test_mode = False
        if dataset_cfg.get("samples_per_gpu", 1) > 1:
            dataset_cfg.pipeline = replace_ImageToTensor(dataset_cfg.pipeline)
    else:
        raise ValueError("Only dict dataset config is supported in this script.")

    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    target_indices = parse_indices(args.indices)
    summary_rows: List[Dict[str, Any]] = []
    saved = 0
    progress = mmcv.ProgressBar(min(len(dataset), args.max_samples))

    for idx, data in enumerate(data_loader):
        if target_indices is not None and idx not in target_indices:
            continue
        if saved >= args.max_samples:
            break

        img = maybe_unwrap_data_container(data.get("img"))
        img_metas = maybe_unwrap_data_container(data.get("img_metas"))
        gt_seg_raw = maybe_unwrap_data_container(data.get("gt_seg_mask"))
        points = maybe_unwrap_data_container(data.get("points"))

        if img is None or img_metas is None:
            logger.warning("Skip idx=%d because img or img_metas is missing.", idx)
            continue
        gt_seg = stack_batch_tensor(gt_seg_raw)
        if gt_seg is None:
            logger.warning(
                "Skip idx=%d because gt_seg_mask is missing. "
                "Try --split train or add --force-train-mode.",
                idx,
            )
            continue

        with torch.no_grad():
            seg_logits, cur_metas = extract_seg_logits(
                model.module, img, img_metas, points
            )
        if seg_logits is None:
            logger.warning("Skip idx=%d because model output has no seg branch.", idx)
            continue

        pred_prob, pred_bin, gt_bin = prepare_prob_and_gt(
            seg_logits=seg_logits,
            gt_seg=gt_seg,
            channel=args.channel,
            thresh=args.seg_thresh,
        )

        bsz = pred_prob.shape[0]
        for b in range(bsz):
            meta = cur_metas[b]
            sample_name = osp.splitext(osp.basename(meta.get("pts_filename", f"idx_{idx}_{b}.bin")))[0]
            metrics = calc_metrics(pred_bin[b], gt_bin[b])
            panel = make_panel(
                prob=pred_prob[b],
                pred=pred_bin[b],
                gt=gt_bin[b],
                metrics=metrics,
                sample_name=sample_name,
                thresh=args.seg_thresh,
            )

            out_img = osp.join(args.show_dir, f"{idx:06d}_{b}_{sample_name}.png")
            Image.fromarray(panel).save(out_img)

            if args.save_npz:
                out_npz = osp.join(args.show_dir, f"{idx:06d}_{b}_{sample_name}.npz")
                np.savez_compressed(
                    out_npz,
                    prob=pred_prob[b].astype(np.float32),
                    pred=pred_bin[b].astype(np.uint8),
                    gt=gt_bin[b].astype(np.uint8),
                )

            row = {
                "dataset_idx": idx,
                "batch_idx": b,
                "sample_name": sample_name,
                **metrics,
                "image_path": out_img,
            }
            summary_rows.append(row)
            saved += 1
            progress.update()

            if saved >= args.max_samples:
                break

    if len(summary_rows) == 0:
        logger.warning("No samples were saved. Please check dataset split/config.")
        return

    csv_path = osp.join(args.show_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    mean_iou = float(np.mean([r["iou"] for r in summary_rows]))
    mean_f1 = float(np.mean([r["f1"] for r in summary_rows]))
    mean_precision = float(np.mean([r["precision"] for r in summary_rows]))
    mean_recall = float(np.mean([r["recall"] for r in summary_rows]))

    report_path = osp.join(args.show_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"samples: {len(summary_rows)}\n")
        f.write(f"mean_iou: {mean_iou:.6f}\n")
        f.write(f"mean_f1: {mean_f1:.6f}\n")
        f.write(f"mean_precision: {mean_precision:.6f}\n")
        f.write(f"mean_recall: {mean_recall:.6f}\n")

    logger.info("Done. samples=%d", len(summary_rows))
    logger.info(
        "mean_iou=%.4f mean_f1=%.4f mean_precision=%.4f mean_recall=%.4f",
        mean_iou,
        mean_f1,
        mean_precision,
        mean_recall,
    )
    logger.info("summary: %s", csv_path)
    logger.info("report: %s", report_path)


if __name__ == "__main__":
    main()
