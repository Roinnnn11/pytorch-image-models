# MobileNetV3 architecture-aware INT8 search

This experiment keeps ModelOpt as the PTQ/Q-DQ backend, but does not accept
`INT8_DEFAULT_CFG` as the final deployment configuration. It evaluates a
bounded, manually designed set of MobileNetV3 FP16 fallback presets on the same
2,000-image class-balanced subset:

- full INT8
- FP16 stem
- FP16 SE
- FP16 depthwise convolutions
- FP16 blocks 0-1
- FP16 blocks 0-3
- FP16 blocks 0-5
- FP16 blocks 0-5 plus stem

The selector chooses the lowest-cost preset whose subset Top1 drop is within
`TARGET_DROP`. If none meets the constraint, it exports the highest-accuracy
preset so the run still produces a usable result.

Run the complete search, TensorRT build, and ImageNet validation:

```bash
cd pytorch-image-models
PY=/data1/liurongying/miniconda3/envs/deepburst/bin/python \
  bash my_optim/run_mobile_search.sh
```

Optional controls:

```bash
SEARCH_SAMPLES=2000 TARGET_DROP=0.5 PY=/path/to/python \
  bash my_optim/run_mobile_search.sh
```

Main outputs:

```text
my_optim/results/mobilenetv3_large_100/fallback_search.json
my_optim/onnx/mobilenetv3_large_100/*_int8_qdq_search_selected_inline.onnx
my_optim/engines/mobilenetv3_large_100/*_int8_search.engine
my_optim/results/mobilenetv3_large_100/trt_int8_search.json
```

## Two-model experiment scope

ViT already has a greedy architecture-group search in
`my_optim_Trans/run_sensitive_fallback.py`. It tests MLP, attention,
patch-embedding, and classifier-head fallback groups on one balanced subset.
The existing validated ViT result can be reused, or reproduced with:

```bash
cd pytorch-image-models/my_optim_Trans
OPTIM_MODEL=vit_base_patch16_224 \
  /data1/liurongying/miniconda3/envs/deepburst/bin/python \
  run_sensitive_fallback.py --sample-size 2000 --target-drop 0.5
```

Together, the two experiments demonstrate different manually designed search
spaces:

- ViT: Transformer sub-structures such as MLP and attention.
- MobileNetV3: stem, depthwise convolution, SE, and early block ranges.

ModelOpt performs calibration and Q/DQ export for every candidate; the final
INT8/FP16 layer configuration is selected by these search scripts.
