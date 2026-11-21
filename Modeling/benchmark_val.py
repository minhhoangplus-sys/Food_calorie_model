"""Benchmark trained YOLO models on the validation set and export the best weights."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# Allow execution from either repo root (`python Modeling/benchmark_val.py`) or `Modeling/`
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import (
    DEFAULT_CONFIG_PATH,
    inspect_dataset,
    load_config,
    resolve_device,
    resolve_model_specs,
    resolve_repo_path,
    write_resolved_data_yaml,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for validation benchmarking."""
    parser = argparse.ArgumentParser(
        description="Step 4: Benchmark trained YOLO models strictly on the validation set."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to Modeling/config.yaml",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional override path to dataset data.yaml",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model tags to evaluate (e.g. yolov8n yolo11n)",
    )
    parser.add_argument("--device", type=str, default=None, help="Override compute device")
    parser.add_argument(
        "--skip-cpu-latency",
        action="store_true",
        help="Skip single-image CPU latency measurement",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate val split & checkpoint paths without running inference or writing files.",
    )
    return parser.parse_args()


def _import_runtime_deps() -> tuple[Any, Any]:
    """Lazily import torch and Ultralytics YOLO with actionable guidance if missing."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            f"Missing runtime dependency ({exc}). Install packages first:\n"
            "  pip install -r requirements-ds.txt"
        ) from exc
    return torch, YOLO


def load_val_ground_truth(
    val_images: list[Path], val_label_dir: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Load pixel-space ground-truth boxes and categorize val images by item multiplicity."""
    gt_by_image: dict[str, dict[str, Any]] = {}
    counts = {"total": 0, "empty": 0, "single_item": 0, "multi_class": 0, "same_class_x2plus": 0}

    for img_path in val_images:
        with Image.open(img_path) as im:
            w, h = im.size

        lbl_path = val_label_dir / f"{img_path.stem}.txt"
        boxes: list[list[float]] = []
        classes: list[int] = []

        if lbl_path.is_file():
            with lbl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h
                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls_id)

        n_boxes = len(classes)
        n_distinct = len(set(classes))
        if n_boxes == 0:
            category = "empty"
        elif n_boxes == 1:
            category = "single_item"
        elif n_distinct >= 2:
            category = "multi_class"
        else:
            category = "same_class_x2plus"

        counts["total"] += 1
        counts[category] += 1
        gt_by_image[img_path.name] = {
            "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
            "classes": np.array(classes, dtype=np.int64),
            "category": category,
        }

    return gt_by_image, counts


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two (N, 4) and (M, 4) xyxy box arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(
        0.0, boxes_a[:, 3] - boxes_a[:, 1]
    )
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(
        0.0, boxes_b[:, 3] - boxes_b[:, 1]
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 safely."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_conf_threshold(
    preds_by_image: dict[str, dict[str, np.ndarray]],
    gt_by_image: dict[str, dict[str, Any]],
    conf_thresh: float,
    iou_thresh: float,
) -> dict[str, Any]:
    """Perform greedy 1-to-1 class-aware matching at a given confidence threshold."""
    buckets = {
        "all": [0, 0, 0],
        "multi_item": [0, 0, 0],
        "multi_class": [0, 0, 0],
        "same_class_x2plus": [0, 0, 0],
    }

    for img_name, gt in gt_by_image.items():
        gt_boxes = gt["boxes"]
        gt_cls = gt["classes"]
        category = gt["category"]

        pred = preds_by_image[img_name]
        keep = pred["confs"] >= conf_thresh
        p_boxes = pred["boxes"][keep]
        p_cls = pred["classes"][keep]
        p_confs = pred["confs"][keep]

        order = np.argsort(-p_confs)
        p_boxes = p_boxes[order]
        p_cls = p_cls[order]

        matched_gt: set[int] = set()
        tp = 0
        fp = 0

        if len(p_boxes) > 0 and len(gt_boxes) > 0:
            ious = compute_iou_matrix(p_boxes, gt_boxes)
            for p_idx in range(len(p_boxes)):
                same_cls = (gt_cls == p_cls[p_idx])
                best_gt = -1
                best_iou = iou_thresh
                for g_idx in np.where(same_cls)[0]:
                    if int(g_idx) in matched_gt:
                        continue
                    val = float(ious[p_idx, g_idx])
                    if val >= best_iou:
                        best_iou = val
                        best_gt = int(g_idx)
                if best_gt >= 0:
                    matched_gt.add(best_gt)
                    tp += 1
                else:
                    fp += 1
        else:
            fp = len(p_boxes)

        fn = len(gt_boxes) - len(matched_gt)

        buckets["all"][0] += tp
        buckets["all"][1] += fp
        buckets["all"][2] += fn

        if category in ("multi_class", "same_class_x2plus"):
            buckets["multi_item"][0] += tp
            buckets["multi_item"][1] += fp
            buckets["multi_item"][2] += fn
            buckets[category][0] += tp
            buckets[category][1] += fp
            buckets[category][2] += fn

    tp_all, fp_all, fn_all = buckets["all"]
    prec, rec, f1 = _prf(tp_all, fp_all, fn_all)
    m_prec, m_rec, m_f1 = _prf(*buckets["multi_item"])
    _, mc_rec, _ = _prf(*buckets["multi_class"])
    _, sc_rec, _ = _prf(*buckets["same_class_x2plus"])

    n_imgs = max(len(gt_by_image), 1)
    balanced_score = 0.5 * f1 + 0.5 * m_f1

    return {
        "conf_threshold": round(conf_thresh, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp_all,
        "fp": fp_all,
        "fn": fn_all,
        "fp_per_image": round(fp_all / n_imgs, 4),
        "fn_per_image": round(fn_all / n_imgs, 4),
        "multi_precision": round(m_prec, 4),
        "multi_recall": round(m_rec, 4),
        "multi_f1": round(m_f1, 4),
        "multi_class_recall": round(mc_rec, 4),
        "same_class_x2plus_recall": round(sc_rec, 4),
        "balanced_score": round(balanced_score, 4),
    }


def measure_latency(
    weights_path: Path,
    image_paths: list[Path],
    imgsz: int,
    device_str: str,
    warmup: int,
    max_images: int,
) -> dict[str, float]:
    """Measure single-image (batch=1) inference latency in milliseconds."""
    torch, YOLO = _import_runtime_deps()
    model = YOLO(str(weights_path))
    sample = image_paths[:max_images]
    is_cuda = device_str != "cpu" and torch.cuda.is_available()

    for img_p in sample[: min(warmup, len(sample))]:
        model.predict(str(img_p), imgsz=imgsz, device=device_str, verbose=False)
    if is_cuda:
        torch.cuda.synchronize()

    timings_ms: list[float] = []
    for img_p in sample:
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(str(img_p), imgsz=imgsz, device=device_str, verbose=False)
        if is_cuda:
            torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    del model
    if is_cuda:
        torch.cuda.empty_cache()

    arr = np.array(timings_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "n_images": len(sample),
    }


def collect_val_predictions(
    weights_path: Path,
    val_images: list[Path],
    imgsz: int,
    predict_iou: float,
    device_str: str,
    min_conf: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Run inference once over all val images at a low conf floor for threshold sweeping."""
    torch, YOLO = _import_runtime_deps()
    model = YOLO(str(weights_path))
    preds: dict[str, dict[str, np.ndarray]] = {}

    for img_p in val_images:
        res_list = model.predict(
            source=str(img_p),
            imgsz=imgsz,
            conf=min_conf,
            iou=predict_iou,
            device=device_str,
            verbose=False,
        )
        boxes_obj = res_list[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            preds[img_p.name] = {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "confs": np.zeros((0,), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int64),
            }
        else:
            preds[img_p.name] = {
                "boxes": boxes_obj.xyxy.cpu().numpy().astype(np.float32),
                "confs": boxes_obj.conf.cpu().numpy().astype(np.float32),
                "classes": boxes_obj.cls.cpu().numpy().astype(np.int64),
            }

    del model
    if device_str != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def write_markdown_report(
    report_path: Path,
    model_summaries: list[dict[str, Any]],
    sweep_df: pd.DataFrame,
    winner: dict[str, Any],
    val_counts: dict[str, int],
    latency_target_ms: float,
    args_skip_cpu: bool,
) -> None:
    """Write human-readable validation benchmark report for Step 4."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    winner_tag = winner["model"]
    winner_sweep = sweep_df[sweep_df["model"] == winner_tag]

    lines = [
        "# Step 4 Validation Benchmark Report (YOLOv8n vs. YOLO11n)",
        "",
        f"> **Generated (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "> **Evaluation Policy:** Evaluated strictly on the **validation split (`val`)**. "
        "The held-out `test` split was **NOT** touched in this step.",
        "",
        "## 1. Validation Split Composition",
        f"- **Total images:** {val_counts['total']}",
        f"- **Single-item (`1 box`):** {val_counts['single_item']}",
        f"- **Multi-class (`>=2 distinct classes`):** {val_counts['multi_class']}",
        f"- **Same-class multi-instance (`>=2 boxes, 1 class`):** {val_counts['same_class_x2plus']}",
        "",
        "## 2. Model Comparison (Validation Set)",
        "",
        "| Model | Precision | Recall | mAP@50 | mAP@50-95 | Best Conf | Balanced F1 | FP / img |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m in model_summaries:
        lines.append(
            f"| **{m['model']}** | {m['val_precision']:.4f} | {m['val_recall']:.4f} | "
            f"{m['val_map50']:.4f} | **{m['val_map50_95']:.4f}** | "
            f"{m['best_conf']:.2f} | {m['best_balanced_score']:.4f} | {m['best_fp_per_img']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## 3. Single-Image (`batch=1`) Inference Latency (Target `< {latency_target_ms:.0f} ms`)",
            "",
            "| Model | Primary Device | Mean (ms) | Median (ms) | P95 (ms) | CPU Mean (ms) | Meets Target |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for m in model_summaries:
        lat = m["primary_latency"]
        cpu_str = "Skipped" if args_skip_cpu or m["cpu_latency"] is None else f"{m['cpu_latency']['mean_ms']:.2f}"
        meets = "PASS" if lat["mean_ms"] < latency_target_ms else "FAIL"
        lines.append(
            f"| **{m['model']}** | {m['device']} | {lat['mean_ms']:.2f} | "
            f"{lat['median_ms']:.2f} | {lat['p95_ms']:.2f} | {cpu_str} | {meets} |"
        )

    lines.extend(
        [
            "",
            "## 4. Selected Winner",
            f"- **Winning Model:** `{winner['model']}` (`mAP@50-95 = {winner['val_map50_95']:.4f}`, `mAP@50 = {winner['val_map50']:.4f}`)",
            f"- **Recommended Confidence Threshold:** `{winner['best_conf']:.2f}`",
            "- **Exported Checkpoint:** `models/best.pt`",
            "- **Metadata:** `models/model_meta.json`",
            "",
            f"## 5. Confidence Threshold Sweep — `{winner_tag}` (`val` set, IoU >= 0.50)",
            "",
            "| Conf | Precision | Recall | F1 | Multi-Item Recall | Multi-Item F1 | FP / img | FN / img |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, row in winner_sweep.iterrows():
        lines.append(
            f"| {row['conf_threshold']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['multi_recall']:.4f} | {row['multi_f1']:.4f} | "
            f"{row['fp_per_image']:.4f} | {row['fn_per_image']:.4f} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run Step 4 validation benchmarking, latency timing, and model selection."""
    args = parse_args()
    cfg = load_config(args.config)
    b_cfg = cfg["benchmark"]
    if b_cfg.get("split") != "val":
        raise ValueError(
            f"Strict isolation violation: benchmark.split must be 'val', got {b_cfg.get('split')!r}."
        )

    paths_cfg = cfg["paths"]
    t_cfg = cfg["training"]

    data_yaml_path = (
        args.data.resolve() if args.data else resolve_repo_path(paths_cfg["data_yaml"])
    )
    dataset_info = inspect_dataset(data_yaml_path)
    runs_dir = resolve_repo_path(paths_cfg["runs_dir"])
    models_export_dir = resolve_repo_path(paths_cfg["models_export_dir"])
    reports_dir = resolve_repo_path(paths_cfg["reports_dir"])
    model_specs = resolve_model_specs(cfg, args.models)

    gt_by_image, val_counts = load_val_ground_truth(
        dataset_info["val_images"], dataset_info["val_label_dir"]
    )

    print("=== Step 4: Validation Benchmark Plan ===")
    print(f"Benchmark split : {b_cfg['split']} (strictly val-only; test split untouched)")
    print(
        f"Val breakdown   : total={val_counts['total']} | 1_box={val_counts['single_item']} | "
        f"multi_class={val_counts['multi_class']} | same_class_x2+={val_counts['same_class_x2plus']}"
    )
    print(f"Export dir      : {models_export_dir}")
    print(f"Reports dir     : {reports_dir}")

    available: list[tuple[str, Path]] = []
    for tag, _ in model_specs:
        ckpt = runs_dir / f"{tag}_vfn28" / "weights" / "best.pt"
        status = "FOUND" if ckpt.is_file() else "MISSING"
        print(f"  -> {tag}: {ckpt} [{status}]")
        if ckpt.is_file():
            available.append((tag, ckpt))

    if args.dry_run:
        print("\n[dry-run] Validation passed. No inference run and no files written.")
        return

    if not available:
        raise SystemExit(
            "No trained checkpoints found. Run `python Modeling/train_yolo.py` first."
        )

    torch, YOLO = _import_runtime_deps()
    device = resolve_device(args.device if args.device is not None else t_cfg.get("device"))
    resolved_yaml = write_resolved_data_yaml(
        dataset_info, runs_dir.parent / "resolved_data.yaml"
    )

    imgsz = int(t_cfg["imgsz"])
    batch = int(t_cfg["batch"])
    iou_match = float(b_cfg["iou_match_threshold"])
    predict_iou = float(b_cfg["predict_iou"])
    conf_thresholds = [float(c) for c in b_cfg["conf_thresholds"]]
    min_conf_floor = min(0.05, min(conf_thresholds))

    all_sweep_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []

    for tag, ckpt in available:
        print(f"\n--- Benchmarking {tag} ({ckpt}) ---")
        yolo_model = YOLO(str(ckpt))
        val_res = yolo_model.val(
            data=str(resolved_yaml),
            split="val",
            imgsz=imgsz,
            batch=batch,
            device=device,
            plots=True,
            verbose=False,
        )
        val_precision = float(val_res.box.mp)
        val_recall = float(val_res.box.mr)
        val_map50 = float(val_res.box.map50)
        val_map50_95 = float(val_res.box.map)
        del yolo_model
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        primary_latency = measure_latency(
            ckpt,
            dataset_info["val_images"],
            imgsz=imgsz,
            device_str=device,
            warmup=int(b_cfg["latency_warmup"]),
            max_images=int(b_cfg["latency_gpu_images"]),
        )
        cpu_latency = None
        if not args.skip_cpu_latency:
            if device == "cpu":
                cpu_latency = primary_latency
            else:
                cpu_latency = measure_latency(
                    ckpt,
                    dataset_info["val_images"],
                    imgsz=imgsz,
                    device_str="cpu",
                    warmup=min(3, int(b_cfg["latency_warmup"])),
                    max_images=int(b_cfg["latency_cpu_images"]),
                )

        preds = collect_val_predictions(
            ckpt,
            dataset_info["val_images"],
            imgsz=imgsz,
            predict_iou=predict_iou,
            device_str=device,
            min_conf=min_conf_floor,
        )

        model_rows: list[dict[str, Any]] = []
        for conf_t in conf_thresholds:
            metrics = evaluate_conf_threshold(preds, gt_by_image, conf_t, iou_match)
            metrics["model"] = tag
            model_rows.append(metrics)
            all_sweep_rows.append(metrics)

        best_row = max(
            model_rows,
            key=lambda r: (r["balanced_score"], -r["fp_per_image"]),
        )
        model_summaries.append(
            {
                "model": tag,
                "weights_path": ckpt,
                "device": device,
                "val_precision": round(val_precision, 4),
                "val_recall": round(val_recall, 4),
                "val_map50": round(val_map50, 4),
                "val_map50_95": round(val_map50_95, 4),
                "primary_latency": primary_latency,
                "cpu_latency": cpu_latency,
                "best_conf": best_row["conf_threshold"],
                "best_balanced_score": best_row["balanced_score"],
                "best_fp_per_img": best_row["fp_per_image"],
            }
        )

    winner = max(
        model_summaries,
        key=lambda m: (m["val_map50_95"], m["val_map50"], -m["primary_latency"]["mean_ms"]),
    )

    models_export_dir.mkdir(parents=True, exist_ok=True)
    exported_weights = models_export_dir / "best.pt"
    shutil.copy2(winner["weights_path"], exported_weights)

    meta_path = models_export_dir / "model_meta.json"
    meta_payload = {
        "winning_model": winner["model"],
        "source_weights": str(winner["weights_path"]),
        "exported_weights": str(exported_weights),
        "recommended_conf_threshold": winner["best_conf"],
        "imgsz": imgsz,
        "nc": dataset_info["nc"],
        "val_metrics": {
            "precision": winner["val_precision"],
            "recall": winner["val_recall"],
            "map50": winner["val_map50"],
            "map50_95": winner["val_map50_95"],
        },
        "latency_ms": {
            "primary": winner["primary_latency"],
            "cpu": winner["cpu_latency"],
        },
        "evaluated_split": "val",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    reports_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = pd.DataFrame(all_sweep_rows)
    col_order = [
        "model",
        "conf_threshold",
        "precision",
        "recall",
        "f1",
        "multi_precision",
        "multi_recall",
        "multi_f1",
        "multi_class_recall",
        "same_class_x2plus_recall",
        "balanced_score",
        "fp_per_image",
        "fn_per_image",
        "tp",
        "fp",
        "fn",
    ]
    sweep_df = sweep_df[col_order]
    sweep_csv_path = reports_dir / "val_conf_sweep.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)

    report_md_path = reports_dir / "val_benchmark_report.md"
    write_markdown_report(
        report_md_path,
        model_summaries=model_summaries,
        sweep_df=sweep_df,
        winner=winner,
        val_counts=val_counts,
        latency_target_ms=float(b_cfg["latency_target_ms"]),
        args_skip_cpu=args.skip_cpu_latency,
    )

    print("\n=== Step 4 Benchmark Complete ===")
    print(f"Winner         : {winner['model']} (mAP@50-95={winner['val_map50_95']:.4f}, best_conf={winner['best_conf']:.2f})")
    print(f"Exported model : {exported_weights}")
    print(f"Model metadata : {meta_path}")
    print(f"Sweep CSV      : {sweep_csv_path}")
    print(f"Report MD      : {report_md_path}")


if __name__ == "__main__":
    main()
