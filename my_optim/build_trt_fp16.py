"""Stage 4: Build TensorRT FP16 engine from the FP32 ONNX.

trtexec is used to build the engine offline. This script wraps the
command and waits for it to complete, logging all output.

Output: engines/resnet50_fp16.engine
"""
import subprocess
import sys
from pathlib import Path


def main():
    onnx_dir = Path(__file__).parent / "onnx"
    engine_dir = Path(__file__).parent / "engines"
    engine_dir.mkdir(parents=True, exist_ok=True)

    onnx_file = onnx_dir / "resnet50_fp32.onnx"
    engine_file = engine_dir / "resnet50_fp16.engine"

    if not onnx_file.exists():
        print(f"ERROR: ONNX not found at {onnx_file}, run export_onnx_fp32.py first")
        sys.exit(1)

    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={onnx_file}",
        f"--saveEngine={engine_file}",
        "--fp16",
        "--verbose",
        f"--loadInputs=input:{onnx_dir}/resnet50_fp32.onnx.data",
    ]

    # trtexec finds external ONNX data relative to the ONNX file's directory;
    # pass --workspace if needed for old TRT, but TRT 10 uses memPoolSize.
    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={onnx_file}",
        f"--saveEngine={engine_file}",
        "--fp16",
    ]

    print("building TRT FP16 engine...")
    print("cmd:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print("trtexec failed")
        sys.exit(result.returncode)
    print(f"Engine saved: {engine_file}")


if __name__ == "__main__":
    main()
