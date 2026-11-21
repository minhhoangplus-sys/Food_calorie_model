"""Train and compare YOLO nano architectures on the curated 28-class VFN dataset."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Allow execution from either repo root (`python Modeling/train_yolo.py`) or `Modeling/`
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
    """Parse command-line options for YOLO training."""
    parser = argparse.ArgumentParser(
        description="Step 4: Train YOLO models with strict parity hyperparameters."
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
        help="Model tags or checkpoints to train (e.g. yolov8n yolo11n yolo26n)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument("--device", type=str, default=None, help="Override compute device")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/dataset and print planned training runs without training or writing files.",
    )
    return parser.parse_args()


def _reset_seeds(seed: int) -> None:
    """Reset Python, NumPy, and PyTorch random seeds before each model run."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_train_params(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge YAML training config with CLI overrides."""
    t_cfg = dict(cfg.get("training", {}))
    if args.epochs is not None:
        t_cfg["epochs"] = args.epochs
    if args.batch is not None:
        t_cfg["batch"] = args.batch
    if args.imgsz is not None:
        t_cfg["imgsz"] = args.imgsz
    if args.device is not None:
        t_cfg["device"] = args.device
    return t_cfg


def main() -> None:
    """Run Step 4 parity training across configured YOLO models."""
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]

    data_yaml_path = (
        args.data.resolve() if args.data else resolve_repo_path(paths_cfg["data_yaml"])
    )
    dataset_info = inspect_dataset(data_yaml_path)
    runs_dir = resolve_repo_path(paths_cfg["runs_dir"])
    model_specs = resolve_model_specs(cfg, args.models)
    t_cfg = build_train_params(cfg, args)

    print("=== Step 4: YOLO Training Plan ===")
    print(f"Dataset root : {dataset_info['dataset_root']}")
    print(f"Classes (nc) : {dataset_info['nc']}")
    print(f"Train images : {dataset_info['train_count']}")
    print(f"Val images   : {dataset_info['val_count']}")
    print(f"Runs dir     : {runs_dir}")
    print(
        f"Hyperparams  : imgsz={t_cfg['imgsz']}, epochs={t_cfg['epochs']}, "
        f"patience={t_cfg['patience']}, batch={t_cfg['batch']}, "
        f"optimizer={t_cfg['optimizer']}, lr0={t_cfg['lr0']}, seed={t_cfg['seed']}"
    )
    for tag, weights in model_specs:
        run_name = f"{tag}_vfn28"
        print(f"  -> {tag}: init={weights} | output={runs_dir / run_name / 'weights' / 'best.pt'}")

    if args.dry_run:
        print("\n[dry-run] Validation passed. No files written and no training started.")
        return

    torch, YOLO = _import_runtime_deps()
    device = resolve_device(t_cfg.get("device"))
    resolved_yaml = write_resolved_data_yaml(
        dataset_info, runs_dir.parent / "resolved_data.yaml"
    )
    print(f"\nUsing device : {device}")
    print(f"Resolved YAML: {resolved_yaml}")

    summary: list[dict[str, Any]] = []
    for tag, weights in model_specs:
        run_name = f"{tag}_vfn28"
        print(f"\n--- Training {tag} ({weights}) ---")
        _reset_seeds(int(t_cfg["seed"]))

        model = YOLO(weights)
        start_time = time.perf_counter()
        model.train(
            data=str(resolved_yaml),
            imgsz=int(t_cfg["imgsz"]),
            epochs=int(t_cfg["epochs"]),
            patience=int(t_cfg["patience"]),
            batch=int(t_cfg["batch"]),
            seed=int(t_cfg["seed"]),
            deterministic=bool(t_cfg["deterministic"]),
            workers=int(t_cfg["workers"]),
            device=device,
            optimizer=str(t_cfg["optimizer"]),
            lr0=float(t_cfg["lr0"]),
            cos_lr=bool(t_cfg["cos_lr"]),
            fliplr=float(t_cfg["fliplr"]),
            scale=float(t_cfg["scale"]),
            hsv_h=float(t_cfg["hsv_h"]),
            hsv_s=float(t_cfg["hsv_s"]),
            hsv_v=float(t_cfg["hsv_v"]),
            project=str(runs_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
        )
        elapsed_min = (time.perf_counter() - start_time) / 60.0
        best_pt = runs_dir / run_name / "weights" / "best.pt"
        summary.append(
            {
                "model": tag,
                "run_dir": runs_dir / run_name,
                "best_pt": best_pt,
                "elapsed_min": elapsed_min,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Training Summary ===")
    for item in summary:
        print(
            f"{item['model']:10s} | {item['elapsed_min']:6.2f} min | "
            f"best.pt exists={item['best_pt'].is_file()} | {item['best_pt']}"
        )


if __name__ == "__main__":
    main()
