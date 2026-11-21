"""Shared configuration, path resolution, and dataset helpers for Step 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config YAML structure in {path}")
    return data


def resolve_repo_path(path_str: str | Path) -> Path:
    """Resolve a path relative to REPO_ROOT unless already absolute."""
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _list_images(img_dir: Path) -> list[Path]:
    """Return sorted image file paths inside img_dir."""
    return sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _infer_label_dir(dataset_root: Path, img_rel_or_abs: str | Path, split_name: str) -> Path:
    """Infer YOLO label directory matching an images directory."""
    img_path = Path(img_rel_or_abs)
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        candidate = Path(*parts)
        resolved = candidate if candidate.is_absolute() else (dataset_root / candidate)
        return resolved.resolve()
    return (dataset_root / "labels" / split_name).resolve()


def inspect_dataset(data_yaml_path: Path) -> dict[str, Any]:
    """Load data.yaml, resolve relative paths cleanly, and verify train/val directories."""
    yaml_path = Path(data_yaml_path).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset YAML structure in {yaml_path}")

    raw_path = Path(data.get("path", "."))
    # Avoid Ultralytics resolving `path: .` against cwd or global DATASETS_DIR.
    dataset_root = (
        raw_path.resolve() if raw_path.is_absolute() else (yaml_path.parent / raw_path).resolve()
    )

    train_entry = data.get("train", "images/train")
    val_entry = data.get("val", "images/val")
    train_img_dir = (
        Path(train_entry).resolve()
        if Path(train_entry).is_absolute()
        else (dataset_root / train_entry).resolve()
    )
    val_img_dir = (
        Path(val_entry).resolve()
        if Path(val_entry).is_absolute()
        else (dataset_root / val_entry).resolve()
    )

    if not train_img_dir.is_dir():
        raise FileNotFoundError(f"Train image directory not found: {train_img_dir}")
    if not val_img_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {val_img_dir}")

    train_label_dir = _infer_label_dir(dataset_root, train_entry, "train")
    val_label_dir = _infer_label_dir(dataset_root, val_entry, "val")

    train_images = _list_images(train_img_dir)
    val_images = _list_images(val_img_dir)

    names = data.get("names", {})
    nc = int(data.get("nc", len(names)))

    resolved_payload = dict(data)
    resolved_payload["path"] = str(dataset_root)

    return {
        "source_yaml": yaml_path,
        "dataset_root": dataset_root,
        "train_img_dir": train_img_dir,
        "val_img_dir": val_img_dir,
        "train_label_dir": train_label_dir,
        "val_label_dir": val_label_dir,
        "train_images": train_images,
        "val_images": val_images,
        "train_count": len(train_images),
        "val_count": len(val_images),
        "nc": nc,
        "names": names,
        "resolved_payload": resolved_payload,
    }


def write_resolved_data_yaml(dataset_info: dict[str, Any], out_path: Path) -> Path:
    """Write a copy of data.yaml with an absolute `path` so Ultralytics resolves splits cleanly."""
    target = Path(out_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_info["resolved_payload"], f, sort_keys=False)
    return target


def resolve_model_specs(cfg: dict[str, Any], cli_models: list[str] | None) -> list[tuple[str, str]]:
    """Return ordered list of (run_tag, weights_ref) pairs from config or CLI override."""
    cfg_models = cfg.get("models", {})
    if not cli_models:
        if not cfg_models:
            raise ValueError("No models specified in config.yaml under 'models'.")
        return [(str(k), str(v)) for k, v in cfg_models.items()]

    specs: list[tuple[str, str]] = []
    for entry in cli_models:
        if entry in cfg_models:
            specs.append((entry, str(cfg_models[entry])))
            continue
        tag = entry[:-3] if entry.endswith(".pt") else entry
        weights = entry if entry.endswith(".pt") else f"{entry}.pt"
        specs.append((tag, weights))
    return specs


def resolve_device(cfg_device: Any) -> str:
    """Resolve compute device lazily (imports torch only during real execution)."""
    if cfg_device is not None and str(cfg_device).strip() != "":
        return str(cfg_device)
    import torch

    return "0" if torch.cuda.is_available() else "cpu"
