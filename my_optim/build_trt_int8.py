"""Stage 5b (alternative): Build TRT INT8 engine directly using TRT's built-in
entropy calibrator, avoiding the modelopt Q/DQ ONNX export path.

modelopt.torch.quantization.quantize() + torch.onnx.export segfaults in this
environment (torch 2.10 / modelopt 0.41 / no ninja C++ extension). Instead we:
  1. Load the FP32 dynamic-batch ONNX.
  2. Feed 500 calibration images through TRT's IInt8EntropyCalibrator2.
  3. Build and serialize the INT8 engine.

The resulting engine is functionally equivalent to the Q/DQ path: TRT selects
INT8 kernels for all quantizable layers, using the calibrated scale factors.

Output: engines/resnet50_int8.engine
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import tensorrt as trt
import common


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds batches of calibration images to TRT.

    Reads the flat calibration directory (symlinks to val images) and applies
    the same eval preprocessing as the accuracy benchmark so scales are
    representative of the real distribution.
    """

    def __init__(self, calib_dir: Path, batch_size: int = 32, cache_file: str = ""):
        super().__init__()
        transform = common.build_val_transform()
        paths = sorted(calib_dir.glob("*.JPEG"))
        assert paths, f"no JPEG files in {calib_dir}"

        # Pre-load all calibration images as a batched float32 numpy array.
        tensors = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        data = torch.stack(tensors)   # (N, 3, 224, 224)

        self._batches = [
            data[i:i + batch_size].numpy()
            for i in range(0, len(data), batch_size)
        ]
        self._idx = 0
        self._buf = None  # persistent device buffer
        self._cache_file = cache_file
        print(f"calibrator: {len(paths)} images, {len(self._batches)} batches")

    def get_batch_size(self) -> int:
        if not self._batches:
            return 1
        return self._batches[0].shape[0]

    def get_batch(self, names):
        if self._idx >= len(self._batches):
            return None
        batch = self._batches[self._idx].astype(np.float32)
        self._idx += 1
        if self._buf is None or self._buf.shape != batch.shape:
            import ctypes
            nbytes = batch.nbytes
            self._buf = torch.empty(batch.shape, dtype=torch.float32, device="cuda")
        self._buf.copy_(torch.from_numpy(batch))
        return [self._buf.data_ptr()]

    def read_calibration_cache(self):
        if self._cache_file and Path(self._cache_file).exists():
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self._cache_file:
            with open(self._cache_file, "wb") as f:
                f.write(cache)
            print(f"calibration cache saved: {self._cache_file}")


def build_int8_engine(
    onnx_path: Path,
    engine_path: Path,
    calib_dir: Path,
    cache_file: str = "",
):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # weakly-typed allows kINT8 flag
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    ok = parser.parse(onnx_path.read_bytes())
    if not ok:
        for i in range(parser.num_errors):
            print(f"parse error {i}: {parser.get_error(i)}")
        sys.exit(1)
    print(f"ONNX parsed OK: {network.num_layers} layers")

    calibrator = ImageCalibrator(calib_dir, batch_size=32, cache_file=cache_file)

    config = builder.create_builder_config()

    # Optimization profile: min=1, opt=32, max=64
    profile = builder.create_optimization_profile()
    profile.set_shape("input", min=(1, 3, 224, 224), opt=(32, 3, 224, 224),
                      max=(64, 3, 224, 224))
    config.add_optimization_profile(profile)

    # Enable INT8 (+ FP16 as fallback for layers TRT can't quantize).
    if not builder.platform_has_fast_int8:
        print("WARNING: platform does not advertise fast INT8")
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    print(f"building INT8 engine -> {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: build_serialized_network returned None")
        sys.exit(1)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"engine saved: {engine_path} ({engine_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    onnx_path = common.ONNX_DIR / "resnet50_fp32_dynbatch_inline.onnx"
    engine_path = common.ENGINE_DIR / "resnet50_int8.engine"
    cache_file = str(common.ENGINE_DIR / "resnet50_int8.cache")

    build_int8_engine(onnx_path, engine_path, common.CALIB_DIR, cache_file)
