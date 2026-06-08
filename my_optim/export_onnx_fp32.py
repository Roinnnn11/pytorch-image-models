"""Stage 3: Export FP32 ONNX for the selected model (OPTIM_MODEL env var).

Uses timm exportable=True to disable non-traceable ops, then exports via
torch.onnx.export with opset 17 and dynamo=True (Torch 2.x dynamo exporter).

Outputs (under onnx/<model>/):
  <model>_fp32.onnx                 (dynamo, fixed batch=1, external data)
  <model>_fp32_inline.onnx          (single-file copy of the above)
  <model>_fp32_dynbatch.onnx        (legacy exporter, dynamic batch axis)
  <model>_fp32_dynbatch_inline.onnx (single-file, TRT-ready, used by FP16 build)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import common


def main():
    common.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_p = str(common.onnx_path("fp32.onnx"))

    print(f"model: {common.MODEL_NAME}  input_size: {common.INPUT_SIZE}")
    print("building model (exportable=True)...")
    model = common.build_model(exportable=True)

    dummy = torch.randn(1, *common.INPUT_SIZE, device=common.DEVICE)

    print(f"exporting -> {onnx_p}")
    torch.onnx.export(
        model,
        dummy,
        onnx_p,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=True,
    )

    import onnx
    m = onnx.load(onnx_p, load_external_data=True)
    onnx.checker.check_model(m)
    print(f"ONNX check OK | nodes={len(m.graph.node)} | output: {onnx_p}")

    inline_p = str(common.onnx_path("fp32_inline.onnx"))
    onnx.save(m, inline_p)
    print(f"Inline copy: {inline_p} ({os.path.getsize(inline_p)/1e6:.1f} MB)")

    # Dynamic-batch ONNX for the TRT FP16 engine. The dynamo exporter bakes in a
    # fixed shape, so set_input_shape(bs=64) would silently process only 1 sample.
    # The legacy TorchScript exporter (dynamo=False) honours dynamic_axes, giving
    # TRT a real dynamic batch dimension to attach an optimization profile to.
    dyn_p = str(common.onnx_path("fp32_dynbatch.onnx"))
    dummy_cpu = torch.randn(1, *common.INPUT_SIZE)
    print(f"\nexporting dynamic-batch ONNX -> {dyn_p}")
    torch.onnx.export(
        model.cpu(),
        dummy_cpu,
        dyn_p,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    m_dyn = onnx.load(dyn_p, load_external_data=True)
    print(f"dynbatch ONNX loaded | nodes={len(m_dyn.graph.node)}")

    dyn_inline_p = str(common.onnx_path("fp32_dynbatch_inline.onnx"))
    onnx.save(m_dyn, dyn_inline_p)
    print(f"dynbatch inline (TRT-ready): {dyn_inline_p} "
          f"({os.path.getsize(dyn_inline_p)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
