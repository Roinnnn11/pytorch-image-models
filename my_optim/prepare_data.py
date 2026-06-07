"""Prepare standard ImageNet validation set as an ImageFolder for timm.

Steps:
  1. Extract ILSVRC2012_img_val.tar (50000 flat JPEGs) into a temp dir.
  2. Map each ILSVRC2012_val_XXXXXXXX.JPEG to its wnid using meta.bin's val
     ordering, then move it into val/{wnid}/.
  3. Sample a fixed-seed calibration subset (symlinks) for INT8 PTQ.

The class index used by timm/torchvision is the sorted-wnid order, so this
ImageFolder yields labels consistent with the pretrained classifier.
"""
import argparse
import os
import random
import tarfile
from pathlib import Path

import torch


VAL_TAR = "/data1/datasets/ILSVRC2012_img_val.tar"
META_BIN = "/data1/datasets/imagenet/meta.bin"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--calib-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    val_dir = out_dir / "val"
    calib_dir = out_dir / "calib"
    tmp_dir = out_dir / "_val_flat"

    # meta.bin = (wnid -> class names, [wnid per val image in filename order])
    wnid_to_classes, val_wnids = torch.load(META_BIN, weights_only=False)
    assert len(val_wnids) == 50000, f"unexpected val list len {len(val_wnids)}"

    # Pre-create class subdirs.
    val_dir.mkdir(parents=True, exist_ok=True)
    for wnid in wnid_to_classes:
        (val_dir / wnid).mkdir(exist_ok=True)

    # Already organized? (count leaf images)
    existing = sum(1 for _ in val_dir.glob("n*/*.JPEG"))
    if existing >= 50000:
        print(f"val already organized ({existing} images), skipping extract/move")
    else:
        print(f"extracting {VAL_TAR} -> {tmp_dir}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(VAL_TAR) as tf:
            tf.extractall(tmp_dir)

        # Files are ILSVRC2012_val_00000001.JPEG .. 00050000; index i (1-based)
        # maps to val_wnids[i-1].
        moved = 0
        for i, wnid in enumerate(val_wnids, start=1):
            fname = f"ILSVRC2012_val_{i:08d}.JPEG"
            src = tmp_dir / fname
            dst = val_dir / wnid / fname
            if src.exists():
                src.rename(dst)
                moved += 1
            elif not dst.exists():
                raise FileNotFoundError(f"missing {src}")
        print(f"moved {moved} images into {val_dir}")
        # Clean up the (now empty) temp dir.
        try:
            tmp_dir.rmdir()
        except OSError:
            print(f"note: {tmp_dir} not empty, leaving in place")

    # Build a fixed-seed calibration subset as symlinks (flat, no labels needed).
    if calib_dir.exists():
        for p in calib_dir.glob("*.JPEG"):
            p.unlink()
    calib_dir.mkdir(parents=True, exist_ok=True)
    all_imgs = sorted(val_dir.glob("n*/*.JPEG"))
    rng = random.Random(args.seed)
    calib_imgs = rng.sample(all_imgs, args.calib_size)
    for p in calib_imgs:
        (calib_dir / p.name).symlink_to(p.resolve())
    print(f"calibration subset: {len(calib_imgs)} images -> {calib_dir} (seed={args.seed})")

    n_val = sum(1 for _ in val_dir.glob("n*/*.JPEG"))
    print(f"DONE. val images: {n_val}, classes: {sum(1 for _ in val_dir.glob('n*'))}")


if __name__ == "__main__":
    main()
