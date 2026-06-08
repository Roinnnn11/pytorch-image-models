"""Shared utilities for the quantization workflow.

Centralizes the model spec, data config, loaders, accuracy eval and latency
measurement so every stage (FP32 baseline, Torch FP16, ONNX/TRT) uses an
identical preprocessing pipeline. This is what makes the cross-precision
comparison fair.

Model is selected via the OPTIM_MODEL env var (default: resnet50). All
artifacts are written under per-model subdirectories so multiple models can
coexist:  onnx/<model>/  engines/<model>/  results/<model>/
The val/calib datasets are model-agnostic and shared at data/.
"""
import os
import sys
import time
from pathlib import Path

# timm is installed in editable mode one level up.
_repo_root = str(Path(__file__).parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import timm
from timm.data import create_transform, resolve_data_config
from timm.data.readers.class_map import load_class_map
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Use the mirror; huggingface.co is unreachable from this host.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Model is parameterized via OPTIM_MODEL; everything else derives from it.
MODEL_NAME = os.environ.get("OPTIM_MODEL", "resnet50")

_BASE = Path(__file__).parent
DATA_ROOT = _BASE / "data"          # shared across models (val + calib)
VAL_DIR = DATA_ROOT / "val"
CALIB_DIR = DATA_ROOT / "calib"
# Per-model artifact directories.
RESULTS_DIR = _BASE / "results" / MODEL_NAME
ONNX_DIR = _BASE / "onnx" / MODEL_NAME
ENGINE_DIR = _BASE / "engines" / MODEL_NAME

DEVICE = "cuda"

# Resolved lazily from the model's pretrained cfg (input size varies by model).
_INPUT_SIZE = None


def _resolve_input_size():
    global _INPUT_SIZE
    if _INPUT_SIZE is None:
        cfg = get_data_config()
        _INPUT_SIZE = tuple(cfg["input_size"])  # (3, H, W)
    return _INPUT_SIZE


def get_input_size():
    """(C, H, W) input size for the selected model, from its pretrained cfg."""
    return _resolve_input_size()


def onnx_path(suffix: str) -> Path:
    """Model-prefixed ONNX path, e.g. onnx_path('fp32_dynbatch_inline.onnx')."""
    return ONNX_DIR / f"{MODEL_NAME}_{suffix}"


def engine_path(suffix: str) -> Path:
    """Model-prefixed engine path, e.g. engine_path('fp16.engine')."""
    return ENGINE_DIR / f"{MODEL_NAME}_{suffix}"


def build_model(exportable: bool = False):
    """Create a pretrained, eval-mode model on CUDA."""
    model = timm.create_model(MODEL_NAME, pretrained=True, exportable=exportable)
    model.eval().to(DEVICE)
    return model


def get_data_config(model=None):
    if model is None:
        # pretrained_cfg (input_size/mean/std/crop_pct) is populated by the
        # registry at create_model time and needs no weight download.
        model = timm.create_model(MODEL_NAME, pretrained=False)
    return resolve_data_config({}, model=model)


def build_val_transform(cfg=None):
    """Eval transform: resize/center-crop/normalize, no random aug."""
    if cfg is None:
        cfg = get_data_config()
    return create_transform(
        input_size=cfg["input_size"],
        interpolation=cfg["interpolation"],
        mean=cfg["mean"],
        std=cfg["std"],
        crop_pct=cfg["crop_pct"],
        crop_mode=cfg.get("crop_mode", "center"),
        is_training=False,
    )


def build_val_loader(batch_size: int = 64, workers: int = 8, cfg=None):
    transform = build_val_transform(cfg)
    dataset = ImageFolder(str(VAL_DIR), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, dataset


def build_calib_loader(batch_size: int = 32, workers: int = 8, cfg=None):
    """Flat calibration folder (symlinks), same eval transform, no labels used."""
    transform = build_val_transform(cfg)
    # ImageFolder needs class subdirs; calib is flat, so wrap in a trivial reader.
    from torchvision.datasets import DatasetFolder
    from PIL import Image

    paths = sorted(CALIB_DIR.glob("*.JPEG"))

    class FlatDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            img = Image.open(paths[i]).convert("RGB")
            return transform(img), 0

    loader = DataLoader(
        FlatDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return loader


@torch.inference_mode()
def evaluate_accuracy(infer_fn, loader, max_batches=None, desc="eval"):
    """Run top1/top5 over loader. infer_fn(images_tensor) -> logits tensor.

    Returns dict with top1, top5, n. infer_fn handles its own dtype/device.
    """
    top1 = top5 = n = 0
    t0 = time.time()
    for bi, (images, targets) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        logits = infer_fn(images)
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits, device=DEVICE)
        _, pred = logits.topk(5, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        n += targets.size(0)
        if bi % 50 == 0:
            print(f"[{desc}] batch {bi} n={n} top1={100*top1/n:.2f}%", flush=True)
    dt = time.time() - t0
    return {
        "top1": 100.0 * top1 / n,
        "top5": 100.0 * top5 / n,
        "n": n,
        "eval_seconds": dt,
    }


@torch.inference_mode()
def measure_latency(infer_fn, batch_size: int, n_warmup=30, n_iter=100):
    """Measure per-batch latency (ms) and throughput (img/s) with CUDA events."""
    x = torch.randn(batch_size, *get_input_size(), device=DEVICE)
    for _ in range(n_warmup):
        infer_fn(x)
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_iter):
        starter.record()
        infer_fn(x)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms
    times.sort()
    mean = sum(times) / len(times)
    median = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]
    return {
        "batch_size": batch_size,
        "latency_ms_mean": mean,
        "latency_ms_median": median,
        "latency_ms_p99": p99,
        "throughput_img_s": 1000.0 * batch_size / mean,
    }


def gpu_name():
    return torch.cuda.get_device_name(0)


# Initialize module-level INPUT_SIZE now that all functions are defined.
# This lets existing `common.INPUT_SIZE` references keep working.
INPUT_SIZE = _resolve_input_size()
