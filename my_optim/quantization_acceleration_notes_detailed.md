# pytorch-image-models 量化加速工作笔记（详细解释版）

## 1. 工作目标

本次工作围绕 `pytorch-image-models` / `timm` 中的典型 ImageNet 分类模型，完成从 PyTorch 推理到 TensorRT 部署的量化加速验证。核心目标包括：

1. 建立公平的 FP32、Torch FP16、TensorRT FP16、TensorRT INT8 对比流程。
2. 使用 ImageNet validation 50k 数据评估精度，用 CUDA event 统计延迟和吞吐。
3. 使用 NVIDIA ModelOpt 做 PTQ 后训练量化，并导出 Q/DQ ONNX。
4. 使用 TensorRT 构建 FP16 / INT8 engine，比较不同模型的实际加速效果。
5. 分析 INT8 没有显著快过 FP16 的原因，为后续继续优化 INT8 提供依据。

本次覆盖的模型包括：

- `resnet50`
- `mobilenetv3_large_100`
- `vit_base_patch16_224`
- `swin_tiny_patch4_window7_224`

测试 GPU 为 `NVIDIA GeForce RTX 3080 Ti`。

## 2. 本次用到的核心技术

这一部分是汇报时最应该重点讲的内容。结果数字只是验证，真正的工作量在于把一条可复现的推理优化 pipeline 搭起来，并在 INT8 不符合预期时定位原因。

### 2.1 timm 模型构建与统一预处理

代码位置：`common.py`

使用的技术点：

- `timm.create_model(model_name, pretrained=True, exportable=True)`
  - `exportable=True` 用于关闭部分不利于 ONNX trace/export 的实现。
  - 不同模型使用同一套构建入口，便于比较 CNN 和 Transformer。

- `resolve_data_config`
  - 自动读取模型对应的 input size、mean、std、crop pct、interpolation。
  - 避免不同模型因为预处理不一致导致精度对比不公平。

- `OPTIM_MODEL` 环境变量
  - 通过 `OPTIM_MODEL=resnet50` 或 `OPTIM_MODEL=vit_base_patch16_224` 切换模型。
  - 产物按模型分目录保存，避免 ONNX、engine、result 互相覆盖。

### 2.2 Baseline 与统一评估

代码位置：`run_baseline.py`、`run_trt_eval.py`、`run_summary.py`

使用的技术点：

- PyTorch FP32 baseline。
- PyTorch autocast FP16 baseline。
- TensorRT engine 推理封装。
- ImageNet val 50k 全量 Top1 / Top5 评估。
- CUDA event 统计 latency。
- batch size 1 代表低延迟场景。
- batch size 32 / 64 代表吞吐场景。
- logit similarity：
  - `MSE`
  - `cosine similarity`
  - 用于判断 engine 输出和 FP32 输出的数值偏差。

这一步的价值是建立统一评估基准。没有统一 baseline，后面的量化结果无法判断是模型本身差异、预处理差异，还是部署后端差异。

### 2.3 ONNX 导出

代码位置：`export_onnx_fp32.py`

使用的技术点：

- `torch.onnx.export`
- opset 17
- dynamic batch axis
- inline ONNX 保存

这里做了两类 ONNX：

1. 普通 FP32 ONNX
   - 用于检查导出正确性。

2. dynamic-batch FP32 ONNX
   - 用于 TensorRT optimization profile。
   - 支持 `min / opt / max` batch size。

一个重要细节是：dynamo exporter 在某些情况下会把 batch 固定住，所以我们额外用 legacy exporter + `dynamic_axes` 导出真正的 dynamic batch ONNX。

### 2.4 TensorRT FP16

代码位置：`build_trt_engine.py`

使用的技术点：

- TensorRT Python API
- `BuilderFlag.FP16`
- optimization profile
- serialized engine

FP16 路线比较直接：

```text
FP32 dynamic-batch ONNX
  -> TensorRT builder
  -> FP16 flag
  -> TRT FP16 engine
```

这条路线是本次最稳定的加速方案，也作为后续 INT8 的强 baseline。

### 2.5 ModelOpt PTQ 与 Q/DQ ONNX

代码位置：`run_ptq_int8.py`

使用的技术点：

- NVIDIA ModelOpt
- PTQ 后训练量化
- `mtq.INT8_DEFAULT_CFG`
- Q/DQ ONNX
- calibration loop

默认配置的含义：

- weight：INT8 per-channel，`axis=0`
- activation：INT8 per-tensor，`axis=None`
- calibration：默认 `max`

核心流程：

```text
PyTorch model
  -> ModelOpt mtq.quantize
  -> fake quantized model
  -> torch.onnx.export
  -> Q/DQ ONNX
```

Q/DQ ONNX 的特点是图里显式出现：

- `QuantizeLinear`
- `DequantizeLinear`

这和 `/data1/liurongying/workplace/optim` 的 Q/DQ 路线是一类方法。

### 2.6 INT8 校准策略探索

代码位置：`my_optim/run_ptq_int8.py`

除了默认 max calibration，还尝试了：

- percentile calibration
  - 通过 HistogramCalibrator 重新收集 activation histogram。
  - 用 percentile 截断离群值。
  - 主要针对 MobileNetV3 中 hard-swish / depthwise conv 造成的 activation outlier。

- mse calibration
  - 尝试降低量化前后重构误差。

- smoothquant
  - 尝试把 activation 的量化难度迁移到 weight。

这些策略说明我们不是只跑默认 PTQ，而是针对精度下降问题做了校准消融。

### 2.7 INT8 TensorRT build 策略

代码位置：`build_trt_engine.py`

这里是整个探索中最关键的技术点。

最初 INT8 v1 使用：

```text
STRONGLY_TYPED + Q/DQ ONNX
```

含义是：TensorRT 根据 ONNX 图中显式的 Q/DQ 节点决定类型和 scale。

后来我们对比 `/optim` 后发现，参考项目使用的是更接近：

```text
--fp16 --int8
```

也就是：

```text
INT8 quantized layers + FP16 fallback for non-INT8 layers
```

因此补做了 INT8 v2：

- weakly typed network
- FP16 flag
- INT8 flag
- 更高 builder optimization level
- timing cache
- 更合适的 optimization profile

这个探索证明：INT8 是否快，不只取决于有没有 Q/DQ，也强依赖 TensorRT build 配置。

### 2.8 敏感层 FP16 fallback

代码位置：`run_sensitive_fallback.py`

使用的技术点：

- ModelOpt quant config override
- 根据 module name fragment 关闭部分 quantizer
- 构建混合精度 Q/DQ ONNX

探索过的回退对象包括：

- `head`
- `attn.qkv`
- `attn.proj`
- `mlp.fc1`
- `mlp.fc2`

ViT 的实验说明 MLP 层是精度敏感区域：把 `mlp.fc1` 和 `mlp.fc2` 回退 FP16 可以显著恢复精度，但速度会下降。这说明混合精度是精度和速度之间的折中手段。

## 3. 探索流程

量化加速流程被拆成多个 stage，每一步都生成独立产物，便于复现和排查：

1. `run_baseline.py`
   - 跑 PyTorch FP32 accuracy / latency。
   - 跑 Torch autocast FP16 accuracy / latency。
   - 保存 `results/<model>/baseline.json`。

2. `export_onnx_fp32.py`
   - 用 `timm.create_model(..., exportable=True)` 构建可导出的模型。
   - 导出 FP32 ONNX。
   - 额外导出 dynamic batch ONNX，供 TensorRT optimization profile 使用。

3. `build_trt_engine.py --precision fp16`
   - 使用 FP32 dynamic-batch ONNX。
   - TensorRT 开启 `BuilderFlag.FP16`。
   - 生成 TensorRT FP16 engine。

4. `run_trt_eval.py --precision fp16`
   - 读取 FP16 engine。
   - 在 ImageNet val 50k 上评估 Top1 / Top5。
   - 统计 batch size 1 和吞吐 batch 的延迟。

5. `run_ptq_int8.py`
   - 使用 ModelOpt 做 PTQ。
   - 默认配置是 `mtq.INT8_DEFAULT_CFG`。
   - 导出带 `QuantizeLinear / DequantizeLinear` 的 Q/DQ ONNX。

6. `build_trt_engine.py --precision int8`
   - 使用 Q/DQ ONNX 构建 TensorRT INT8 engine。
   - INT8 v1 采用 `STRONGLY_TYPED` 解析 Q/DQ 图。
   - 后续 INT8 v2 改为 `FP16 + INT8` build，用于验证更接近 `/optim` 的构建策略。

7. `run_trt_eval.py --precision int8`
   - 读取 INT8 engine。
   - 在 ImageNet val 50k 上评估精度和速度。

8. `run_summary.py`
   - 汇总 FP32、Torch FP16、TRT FP16、TRT INT8。
   - 计算相对 FP32 的精度下降、延迟加速比、logit MSE 和 cosine similarity。

从探索角度看，整个过程可以概括为：

1. 先建立 FP32 / Torch FP16 baseline。
   - 目的：确定 PyTorch 原始精度和基础速度。

2. 再完成 ONNX export 和 TensorRT FP16。
   - 目的：建立强 baseline。
   - 结果：TRT FP16 稳定、精度几乎无损。

3. 然后接入 ModelOpt PTQ Q/DQ INT8。
   - 目的：验证 INT8 pipeline 能否跑通。
   - 产物：Q/DQ ONNX 和 INT8 engine。

4. 发现 INT8 v1 不符合预期。
   - 有些模型 INT8 比 FP16 慢。
   - 有些模型精度下降明显。

5. 验证 ONNX 是否真的有 Q/DQ。
   - 检查 `QuantizeLinear / DequantizeLinear` 节点数量。
   - 结论：Q/DQ 是存在的，问题不是“没量化”。

6. 对比 `/optim` 的 Q/DQ 方法。
   - 发现 `/optim` 的 INT8 build 实际是 `--fp16 --int8`。
   - 推断 v1 的 strongly typed build 可能缺少 FP16 fallback。

7. 补做 INT8 v2。
   - 使用 `FP16 + INT8` build。
   - 使用更激进的 builder optimization level 和 timing cache。
   - ViT / Swin 速度明显提升。

8. 针对精度问题做校准和敏感层分析。
   - MobileNetV3：percentile / mse calibration。
   - ViT：attention/head/MLP FP16 fallback。

这个流程体现的是“先跑通，再验证，再定位，再修正”的实验路线。

## 4. 目录与脚本组织

### `my_optim`

主要用于 CNN 模型：

- `resnet50`
- `mobilenetv3_large_100`

关键脚本：

- `common.py`：统一模型名、数据路径、预处理、accuracy eval、latency 测量。
- `run_baseline.py`：PyTorch FP32 / FP16 baseline。
- `export_onnx_fp32.py`：导出 FP32 ONNX 和 dynamic-batch ONNX。
- `run_ptq_int8.py`：ModelOpt PTQ + Q/DQ ONNX 导出。
- `build_trt_engine.py`：构建 TensorRT FP16 / INT8 engine。
- `run_trt_eval.py`：TensorRT engine accuracy / latency eval。
- `run_summary.py`：结果汇总。

### `my_optim_Trans`

主要用于 Transformer / attention-based 模型：

- `vit_base_patch16_224`
- `swin_tiny_patch4_window7_224`

相比 `my_optim`，它把吞吐 batch size 控制在 32，以适配 12GB 显存，并增加了：

- `run_all.sh`：一键运行 ViT / Swin 全流程。
- `run_sensitive_fallback.py`：INT8 精度下降较大时，尝试把敏感层回退到 FP16。

## 5. 量化方法说明

本次 INT8 不是 TensorRT entropy calibrator 直接校准 FP32 ONNX，而是 ModelOpt Q/DQ 路线：

```text
PyTorch model
  -> ModelOpt PTQ
  -> fake quantized PyTorch model
  -> export Q/DQ ONNX
  -> TensorRT parses QuantizeLinear / DequantizeLinear
  -> TensorRT INT8 engine
```

默认量化配置：

```python
mtq.INT8_DEFAULT_CFG
```

该配置的核心含义：

- weight quantizer：8 bit，`axis=0`，即权重 per-channel。
- input activation quantizer：8 bit，`axis=None`，即 activation per-tensor。
- 默认校准算法：`max`。
- BatchNorm、LeakyReLU 以及一些输出层默认不量化。

因此，本次量化属于：

```text
PTQ + Q/DQ ONNX + TensorRT INT8
```

这和 `/data1/liurongying/workplace/optim` 项目中的 Q/DQ 思路是一类方法。区别主要不在“有没有 QDQ”，而在 TensorRT engine 构建策略。

## 6. Q/DQ 是否真的存在

我检查了导出的 ONNX 节点，确认每个 INT8 ONNX 中都有 `QuantizeLinear` 和 `DequantizeLinear`。

| 模型 | QuantizeLinear | DequantizeLinear |
|---|---:|---:|
| ResNet50 | 110 | 110 |
| MobileNetV3-Large | 127 | 127 |
| ViT-B/16 | 100 | 100 |
| Swin-Tiny | 106 | 106 |
| `/optim` DeepBurst 参考项目 | 96 | 96 |

这说明当前 INT8 产物确实是 Q/DQ 图，不是单纯文件名叫 int8。

## 7. 和 `/optim` Q/DQ 实现的关键差异

`/optim` 的路线也是：

```text
ModelOpt quantize -> Q/DQ ONNX -> TensorRT INT8
```

但是它通过 ModelOpt deploy/runtime 的 builder 构建 engine：

```python
build_engine(..., trt_mode=TRTMode.INT8)
```

ModelOpt 内部的 `TRTMode.INT8` 对应 trtexec 参数：

```text
--fp16 --int8
```

也就是说，`/optim` 更接近：

```text
INT8 quantized layers + FP16 fallback for non-INT8 layers
```

而本项目早期主流程的 `build_trt_engine.py` 对 INT8 使用：

```python
network_flags = STRONGLY_TYPED
```

并且没有设置：

```python
BuilderFlag.INT8
BuilderFlag.FP16
```

这样做可以让 TensorRT 按 Q/DQ 图的显式类型构建 engine，但它也带来一个问题：没有被 Q/DQ 覆盖的算子可能不是 FP16 fallback，而是继续以 FP32 运行。对 Transformer 来说，LayerNorm、Softmax、Transpose、Reshape、Elementwise 等非量化算子很多，这会显著拖慢 INT8 engine。

因此，早期 INT8 v1 速度不理想，不能简单理解为“INT8 没有效果”，更准确地说是：

```text
INT8 v1 的 Q/DQ build 配置还没有达到最佳部署形态。
```

后续补做的 INT8 v2 改成 `FP16 + INT8` build 后，ViT 和 Swin 的速度明显提升，这进一步证明问题主要出在 build 策略，而不是 Q/DQ 方法本身。

## 8. 实验结果汇总

说明：`summary.json` 中的主结果最初记录的是 INT8 v1，即 `STRONGLY_TYPED` Q/DQ build。后续又补做了 INT8 v2：`weakly-typed + FP16 + INT8 flags`，相当于 trtexec `--fp16 --int8`，并提高 builder optimization level、使用 timing cache。ViT 和 Swin 的最新 INT8 结论应以 v2 为准。

### 8.1 ResNet50

batch 吞吐使用 `bs=64`。

| 精度模式 | Top1 | Top5 | Top1 drop | bs1 延迟 | bs64 延迟 | 吞吐 | 相对 FP32 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 torch | 80.376 | 94.590 | 0.000 | 4.074 ms | 48.260 ms | 1326 img/s | 1.00x |
| Torch FP16 | 80.364 | 94.588 | 0.012 | 5.233 ms | 27.792 ms | 2303 img/s | 1.74x |
| TRT FP16 | 80.380 | 94.588 | -0.004 | 1.106 ms | 9.453 ms | 6770 img/s | 5.10x |
| TRT INT8 | 78.764 | 93.804 | 1.612 | 1.167 ms | 14.873 ms | 4303 img/s | 3.24x |

分析：

- TRT FP16 精度基本无损，并且速度最佳。
- TRT INT8 有 1.61% Top1 下降，速度也慢于 TRT FP16。
- 对 ResNet50 来说，当前 INT8 不如 FP16 划算。

### 8.2 MobileNetV3-Large

batch 吞吐使用 `bs=64`。

| 精度模式 | Top1 | Top5 | Top1 drop | bs1 延迟 | bs64 延迟 | 吞吐 | 相对 FP32 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 torch | 75.772 | 92.534 | 0.000 | 4.481 ms | 13.394 ms | 4778 img/s | 1.00x |
| Torch FP16 | 75.768 | 92.538 | 0.004 | 5.807 ms | 8.249 ms | 7758 img/s | 1.62x |
| TRT FP16 | 75.754 | 92.524 | 0.018 | 0.858 ms | 3.320 ms | 19280 img/s | 4.03x |
| TRT INT8 | 71.664 | 90.032 | 4.108 | 0.982 ms | 6.951 ms | 9207 img/s | 1.93x |

分析：

- TRT FP16 速度非常好，精度几乎不变。
- MobileNetV3 的 INT8 精度下降明显，Top1 drop 超过 4%。
- 该模型包含 hard-swish、depthwise conv、SE 等结构，activation 分布对 max calibration 很敏感。
- 虽然做了 percentile / mse 校准消融，但当前 INT8 仍不适合作为最终部署方案。

MobileNetV3 INT8 校准消融：

| 变体 | 校准方式 | Top1 | Top1 变化 | bs64 延迟 | 吞吐 |
|---|---|---:|---:|---:|---:|
| V0 | max | 59.664 | -16.106 | 7.088 ms | 9029 img/s |
| V1 | percentile 99.9 | 70.574 | -5.196 | 13.955 ms | 4586 img/s |
| V2 | percentile 99.99 | 71.664 | -4.106 | 6.951 ms | 9207 img/s |
| V3 | mse | 68.822 | -6.948 | 7.100 ms | 9014 img/s |

percentile 99.99 是当前相对最好的 INT8 变体，但仍然没有达到“准确率不低”的目标。

### 8.3 ViT-B/16

batch 吞吐使用 `bs=32`。

| 精度模式 | Top1 | Top5 | Top1 drop | bs1 延迟 | bs32 延迟 | 吞吐 | 相对 FP32 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 torch | 85.104 | 97.526 | 0.000 | 4.501 ms | 89.782 ms | 356 img/s | 1.00x |
| Torch FP16 | 85.110 | 97.528 | -0.006 | 4.741 ms | 29.432 ms | 1087 img/s | 3.05x |
| TRT FP16 | 85.104 | 97.530 | 0.000 | 3.494 ms | 16.762 ms | 1909 img/s | 5.36x |
| TRT INT8 v1 strongly typed | 83.312 | 96.756 | 1.792 | 2.102 ms | 21.273 ms | 1504 img/s | 4.22x |
| TRT INT8 v2 FP16+INT8 | 83.376 | 96.778 | 1.728 | 1.535 ms | 9.649 ms | 3316 img/s | 9.30x |

分析：

- v1 说明 strongly typed Q/DQ 可以跑通，但 batch throughput 不如 FP16。
- v2 对齐 `--fp16 --int8` 后，bs1 和 bs32 都明显快于 TRT FP16。
- v2 的 Top1 drop 为 1.728%，仍是主要问题。
- ViT 的 MLP 层对 INT8 精度非常敏感。

ViT 额外实验：

1. `WeaklyTyped + FP16 + INT8` build 变体：
   - Top1 = 83.376，比标准 INT8 v1 的 83.312 略好。
   - bs1 = 1.535 ms，bs32 = 9.649 ms，吞吐约 3316 img/s。
   - 说明 build 配置对速度影响很大，但精度改善有限。

2. `head + attn.qkv + attn.proj` 回退 FP16：
   - Top1 = 83.460。
   - bs1 = 2.568 ms，bs32 = 26.986 ms。
   - 精度改善有限，速度下降。

3. `mlp.fc1 + mlp.fc2` 回退 FP16：
   - Top1 = 84.996，距离 FP32 只差约 0.108%。
   - 但 bs1 = 4.308 ms，bs32 = 39.902 ms，速度明显变慢。

结论：

- ViT 的 INT8 精度瓶颈主要来自 MLP。
- 如果追求准确率，MLP 回退 FP16 有效。
- 如果追求速度，MLP 回退会抵消 INT8 加速收益。

### 8.4 Swin-Tiny

batch 吞吐使用 `bs=32`。

| 精度模式 | Top1 | Top5 | Top1 drop | bs1 延迟 | bs32 延迟 | 吞吐 | 相对 FP32 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 torch | 81.378 | 95.540 | 0.000 | 6.626 ms | 46.028 ms | 695 img/s | 1.00x |
| Torch FP16 | 81.380 | 95.540 | -0.002 | 8.757 ms | 22.435 ms | 1426 img/s | 2.05x |
| TRT FP16 | 81.358 | 95.544 | 0.020 | 1.347 ms | 7.957 ms | 4021 img/s | 5.78x |
| TRT INT8 v1 strongly typed | 80.902 | 95.258 | 0.476 | 1.106 ms | 10.387 ms | 3081 img/s | 4.43x |
| TRT INT8 v2 FP16+INT8 | 81.006 | 95.328 | 0.372 | 1.186 ms | 6.056 ms | 5284 img/s | 7.60x |

分析：

- Swin-Tiny 是当前 INT8 最健康的模型。
- v2 Top1 只下降 0.372%。
- v2 的 bs32 吞吐达到 5284 img/s，超过 TRT FP16 的 4021 img/s。
- 这说明此前 INT8 慢于 FP16 的主要原因不是 Q/DQ 方法本身，而是 build 策略没有充分启用 FP16 fallback 和更激进的 TensorRT 优化。

## 9. 总体结论

### 9.1 TRT FP16 是当前最稳的部署方案

四个模型中，TRT FP16 都实现了接近无损的精度，并且有稳定加速：

- ResNet50：5.10x
- MobileNetV3：4.03x
- ViT-B/16：5.36x
- Swin-Tiny：5.78x

因此，如果目标是稳定部署，当前最推荐 TRT FP16。

### 9.2 INT8 当前结论需要区分 v1 和 v2

INT8 v1 结果不符合“比 FP16 快且准确率不低”的期待：

- ResNet50：INT8 比 FP16 慢，Top1 还下降 1.61%。
- MobileNetV3：INT8 比 FP16 慢，Top1 下降 4.11%。
- ViT-B/16 v1：INT8 bs1 快于 FP16，但 batch throughput 慢于 FP16，Top1 下降 1.79%。
- Swin-Tiny v1：INT8 bs1 略快，精度下降可接受，但 batch throughput 慢于 FP16。

补做 v2 后，Transformer 模型的速度结论发生变化：

- ViT-B/16 v2：bs1 1.535 ms，bs32 9.649 ms，吞吐 3316 img/s，速度明显超过 TRT FP16；但 Top1 仍下降 1.728%。
- Swin-Tiny v2：bs1 1.186 ms，bs32 6.056 ms，吞吐 5284 img/s，速度和精度都优于 v1，Top1 只下降 0.372%。

因此，最新结论是：INT8 的速度潜力已经在 v2 中验证出来，但不同模型的精度问题仍需要单独优化。

### 9.3 INT8 不理想的主要原因

1. TensorRT build 方式是否有 FP16 fallback 非常关键。

   INT8 v1 使用 `STRONGLY_TYPED`，没有显式开启 `--fp16 --int8`。这可能导致未量化算子以 FP32 执行，从而拖慢 engine。INT8 v2 改成 `FP16 + INT8` build 后，ViT 和 Swin 的速度明显提升。

2. 非量化算子比例较高。

   Transformer 模型有很多 LayerNorm、Softmax、Transpose、Reshape、Elementwise 等算子。这些算子即使主干 MatMul INT8，也会影响整体吞吐。

3. TRT FP16 baseline 已经很强。

   本次对比对象不是普通 PyTorch FP16，而是 TensorRT FP16。TRT FP16 已经做了大量 fusion 和 Tensor Core 优化，因此 INT8 想再大幅超过 FP16 并不容易。

4. 校准策略仍偏基础。

   默认 PTQ 使用 activation per-tensor max calibration。MobileNetV3 和 ViT 对 activation scale 非常敏感，需要更细粒度的校准、混合精度或 QAT。

5. dynamic shape profile 可能不是最优。

   当前 engine 支持动态 batch，例如 `min=1, opt=32, max=64` 或 Transformer 的 `min=1, opt=16, max=32`。实际测速在 max batch 上跑时，可能没有固定 shape engine 那么极致。

## 10. 后续优化建议

如果继续把 INT8 作为重点，可以按下面顺序推进：

1. 对齐 `/optim` 的 TensorRT build 策略。
   - 使用 Q/DQ ONNX。
   - 用 weakly typed network + `BuilderFlag.FP16` + `BuilderFlag.INT8`。
   - 或直接使用 ModelOpt `build_engine(..., trt_mode=TRTMode.INT8)`。
   - 开 builder optimization level 4 或更高。
   - 使用 timing cache。

2. 重新构建 INT8 engine 后先只看速度。
   - 如果速度恢复，说明主要问题是 build 配置。
   - 如果速度仍慢，再做 layer profiling。

3. 做 layerwise profiling。
   - 看哪些层仍然是 FP32 / FP16。
   - 看 Q/DQ 是否被融合。
   - 看是否有大量 reformat / copy / transpose。

4. 优化校准策略。
   - MobileNetV3：重点处理 hard-swish、SE、depthwise conv 附近 activation。
   - ViT：重点处理 MLP 层 activation scale。
   - Swin：当前精度较健康，可以优先做速度优化。

5. 混合精度策略。
   - 将少数敏感层保留 FP16。
   - 目标是在精度和速度之间找到 Pareto 最优点。

6. 必要时考虑 QAT。
   - 如果 PTQ 始终无法兼顾精度和速度，可以对 MobileNetV3 或 ViT 做量化感知训练。

## 11. 汇报时可以强调的结论

本次工作已经完成了端到端量化加速 pipeline：

- 数据预处理统一。
- 模型导出统一。
- FP32 / FP16 / INT8 评估统一。
- 多模型结果完整。
- 对 INT8 的问题做了校准消融和敏感层回退分析。

最重要的认识是：

```text
TensorRT FP16 是稳定可用的加速方案；
INT8 使用了正确的 Q/DQ 方法，v2 build 已经验证出明显速度潜力；
后续重点是降低不同模型的 INT8 精度损失。
```

因此，当前工作可以作为“量化加速实验阶段完成”的结果；如果后续继续深入，则重点应放在 INT8 engine build 对齐、layer profiling 和更细粒度校准上。
