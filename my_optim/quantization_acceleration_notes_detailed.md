# pytorch-image-models 量化加速工作笔记（详细解释版）

> **2026-06-15 四模型复测完成**
>
> 当前代码已经修正 TensorRT 重复同步、Transformer fallback 非代表性抽样、
> ModelOpt 排除规则，并把 FP16+INT8、opt level 5、timing cache 等新版
> builder 配置同步到 CNN。ResNet50、MobileNetV3-Large、ViT-B/16 和 Swin-T
> 均已在原环境完成新版评测。第 8 节之后为当前正式结果；旧数据仅作为实验
> 演进记录，不再用于最终横向比较。

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

后续补做的 INT8 v2 改成 `FP16 + INT8` build 后，四个模型均使用新版流程
重新评测。Swin 的 v2 吞吐达到旧 v1 的 2.12x，说明早期速度问题很大一部分
来自 builder 配置；不同模型剩余的精度问题则需要结合架构单独处理。

## 8. 实验结果汇总

统一口径：

- 精度使用 ImageNet validation 50k Top1。
- `ΔTop1 = 当前 Top1 - FP32 Top1`，负数表示精度下降，单位为百分点。
- 延迟为 `bs=1`，吞吐统一使用 `bs=32`。
- 除明确标为“旧引擎”的 Swin v1 历史对照外，正式 INT8 结果均使用修正后的
  builder 和评测脚本。

### 8.1 四模型推荐结果

| 模型 | 推荐路径 | Top1 | ΔTop1 | bs1 延迟 | bs32 吞吐 | 推荐理由 |
|---|---|---:|---:|---:|---:|---|
| ViT-B/16 | INT8 混合（MLP→FP16） | 85.034 | −0.070 | 1.87 ms | 2367 img/s | 几乎无损，bs1 较 FP32 加速 2.41x |
| Swin-T | 全量 INT8 v2 | 81.006 | −0.372 | 1.25 ms | 5358 img/s | 无需 fallback，吞吐为 FP16 的 1.33x |
| ResNet50 | TRT FP16 | 80.406 | +0.030 | 1.00 ms | 7831 img/s | 当前混合 INT8 仍下降 1.060 个百分点 |
| MobileNetV3-L | TRT FP16 | 75.762 | +0.006 | 0.639 ms | 20602 img/s | 混合 INT8 收益很小，FP16 更简单 |

这张表是部署建议，不代表其他路径没有实验价值。完整数据如下。

### 8.2 ResNet50

| 精度模式 | Top1 | ΔTop1 | bs1 延迟 | bs32 吞吐 | bs1 加速 | 吞吐加速 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 torch | 80.376 | — | 4.31 ms | 1455 img/s | 1.00x | 1.00x |
| TRT FP16 | 80.406 | +0.030 | 1.00 ms | 7831 img/s | 4.31x | 5.38x |
| TRT INT8 全量 | 78.738 | −1.638 | **0.888 ms** | **10177 img/s** | **4.85x** | **6.99x** |
| TRT INT8 混合（layer1+2→FP16） | 79.316 | −1.060 | 0.917 ms | 9461 img/s | 4.70x | 6.50x |

分析：

- 全量 INT8 的吞吐比 TRT FP16 高 30.0%，但损失 1.638 个百分点。
- 回退 `layer1+2` 后恢复 0.578 个百分点，性能只比全量 INT8 低约 7%，说明
  浅层是本次分组实验中更敏感的区域。
- 但混合精度仍比 FP32 低 1.060 个百分点，尚未达到近乎无损，因此当前正式
  部署仍优先 TRT FP16。
- 早期统计中 pool/maxpool 附近 amax 约为 53.4，与浅层敏感现象一致。它可能
  来自浅层激活离群值或通道间尺度差异，但仍需逐层误差和 profiler 数据确认。
- 下一步优先尝试更小粒度的浅层回退、percentile/MSE 校准或 cross-layer
  equalization。activation per-channel 只有在部署后端支持时才值得尝试。

### 8.3 MobileNetV3-Large

| 精度模式 | Top1 | ΔTop1 | bs1 延迟 | bs32 吞吐 | bs1 加速 | 吞吐加速 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 torch | 75.756 | — | 4.68 ms | 4267 img/s | 1.00x | 1.00x |
| TRT FP16 | 75.762 | +0.006 | **0.639 ms** | **20602 img/s** | **7.32x** | **4.83x** |
| TRT INT8 全量 | 59.516 | −16.240 | 0.916 ms | 13181 img/s | 5.11x | 3.09x |
| TRT INT8 混合（blocks 0–5→FP16） | 75.658 | −0.098 | 0.662 ms | 19812 img/s | 7.07x | 4.64x |

分析：

- 全量 per-tensor INT8 下降 16.240 个百分点，且速度低于 TRT FP16，不能用于
  最终部署。
- 回退 blocks 0–5 后恢复 **16.142 个百分点**，最终只比 FP32 低 0.098 个
  百分点，并保留 TRT FP16 **96.2%** 的吞吐。
- 但该方案只让尾部少量 block 保持 INT8，实际 INT8 覆盖率和额外工程复杂度
  不成比例。因此默认推荐 TRT FP16；只有必须保留 INT8 部署链路时才使用混合
  engine。
- blocks 0–5 包含 depthwise convolution、hard-swish 和 SE。观察到
  `blocks.0.0` 输出 amax 约 82.4，blocks.2 的 SE 输出也更宽，说明早期通道
  尺度不均和离群值可能是 per-tensor scale 失真的主要机制。

旧流程中的校准消融仍可作为补充证据，但不能替代新版最终结果：

| 旧变体 | 校准方式 | Top1 | 相对旧 FP32 下降 |
|---|---|---:|---:|
| V0 | max | 59.664 | 16.106 个百分点 |
| V1 | percentile 99.9 | 70.574 | 5.196 个百分点 |
| V2 | percentile 99.99 | 71.664 | 4.106 个百分点 |
| V3 | mse | 68.822 | 6.948 个百分点 |

percentile 99.99 虽比 max calibration 好，但仍不如 blocks 0–5 混合精度。

### 8.4 ViT-B/16

> 修正后正式结果：去除 TRT wrapper 重复同步，并使用 2,000 张类别均衡子集
> 搜索 fallback。以下结果均由修正后的流程重新构建或评测。

| 精度模式 | Top1 | Top1 drop（百分点） | bs1 延迟 | bs32 吞吐 | bs1 加速 | 吞吐加速 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 torch | 85.104 | 0.000 | 4.50 ms | 356 img/s | 1.00x | 1.00x |
| TRT FP16 | 85.104 | 0.000 | 2.95 ms | 2409 img/s | 1.53x | 6.77x |
| TRT INT8 全量 | 83.312 | 1.792 | **1.80 ms** | 1979 img/s | **2.50x** | 5.56x |
| **TRT INT8 混合精度（MLP→FP16）** | **85.034** | **0.070** | **1.87 ms** | **2367 img/s** | **2.41x** | **6.65x** |

分析：

- TRT FP16 完全保持 FP32 精度，bs=1 加速 1.53x，bs=32 吞吐加速 6.77x。
- 全量 INT8 的 bs=1 最快，但 Top1 下降 1.792 个百分点。
- MLP 回退 FP16 后，Top1 恢复 1.722 个百分点，只比 FP32 低 0.070 个百分点。
- 混合精度仅比全量 INT8 慢 0.07 ms，但吞吐反而高 19.6%；它还保留了 TRT
  FP16 98.3% 的吞吐，因此是本次 ViT 实验的最佳精度/速度折中。

类别均衡 fallback 搜索（相同 2,000 张图像）：

| 配置 | 子集 Top1 | 相对 FP32 下降 |
|---|---:|---:|
| FP32 | 84.7 | 0.0 个百分点 |
| 默认 INT8 | 83.2 | 1.5 个百分点 |
| **MLP→FP16** | **84.4** | **0.3 个百分点** |

子集搜索和 ImageNet val 50k 的最终 engine 结果方向一致，确认 MLP
`fc1/fc2` 是本次 ViT-B/16 PTQ 精度损失的主因。这里的“主因”不是只根据
旧的层分布观察推测，而是经过同一批样本公平对照后得到的结果。

> 修正前曾记录 TRT FP16 3.494 ms / 1909 img/s、INT8 v2
> 83.376% / 1.535 ms / 3316 img/s，以及 MLP fallback 84.996%。
> 这些结果受到重复同步或旧 fallback 抽样口径影响，仅作为历史记录。

结论：

- ViT 的 INT8 精度瓶颈主要来自 MLP。
- MLP 回退 FP16 不仅有效恢复精度，而且没有抵消主要加速收益。
- 当前部署优先选择 MLP→FP16 混合精度；只有在 bs=1 延迟比精度更重要时，
  才优先考虑全量 INT8。

### 8.5 Swin-Tiny

| 精度模式 | Top1 | ΔTop1 | bs1 延迟 | bs32 吞吐 | bs1 加速 | 吞吐加速 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 torch | 81.378 | — | 6.63 ms | 695 img/s | 1.00x | 1.00x |
| TRT FP16 | 81.358 | −0.020 | 1.35 ms | 4021 img/s | 4.91x | 5.79x |
| TRT INT8 v1（旧引擎） | 80.902 | −0.476 | 1.29 ms | 2523 img/s | 5.14x | 3.63x |
| TRT INT8 v2 | 81.006 | −0.372 | **1.25 ms** | **5358 img/s** | **5.30x** | **7.71x** |

分析：

- v2 Top1 只下降 0.372 个百分点，当前无需 fallback。
- v2 吞吐为 TRT FP16 的 1.33x、旧 v1 的 2.12x，证明 builder 配置修正有效。
- 在本次模型和校准配置下，Swin 的窗口化层级结构比 ViT 全局注意力表现出
  更好的全量 INT8 适配性；这是实验观察，不应外推为所有窗口模型都如此。

## 9. 总体结论

### 9.1 架构差异决定量化策略

| 模型 | 架构特征 | 全量 INT8 下降 | 混合精度下降 | 本次敏感区域 | 推荐路径 |
|---|---|---:|---:|---|---|
| ViT-B/16 | 全局 attention + MLP | 1.792 | 0.070 | MLP fc1/fc2 | INT8 混合 |
| Swin-T | 层级窗口 attention | 0.372 | 无需 | 未发现必须回退的层组 | 全量 INT8 v2 |
| ResNet50 | 标准残差 CNN | 1.638 | 1.060 | 浅层 layer1/2 | FP16 或继续调校准 |
| MobileNetV3-L | depthwise + SE + hard-swish | 16.240 | 0.098 | blocks 0–5 | FP16；混合作为备选 |

核心规律：

1. **Swin 最适合全量 INT8**：精度下降小，且吞吐明显超过 FP16。
2. **ViT 需要结构化混合精度**：MLP 是主要敏感区域，回退后几乎无损。
3. **ResNet 的敏感区域在浅层**：与“越深越敏感”的直觉不一致，说明必须实测，
   不能只按层深猜测。
4. **MobileNetV3 对 per-tensor activation 最敏感**：depthwise 和 SE 的通道
   尺度差异使全量 INT8 失效，大片回退后才恢复精度。

### 9.2 混合精度有效，但不是自动等于最优

分组 fallback 在 ViT、ResNet 和 MobileNetV3 上都恢复了精度，但当前仓库的
类别均衡自动搜索脚本只覆盖 Transformer。CNN 的回退组仍是实验候选配置，
不能声称已经自动找到“最小 FP16 集合”。

此外，更多 INT8 层并不保证 engine 更快。ViT 和 MobileNetV3 的混合 engine
吞吐都高于其全量 INT8 engine，说明 tactic、fusion、reformat 和不友好的 INT8
kernel 也会决定最终性能。需要 TensorRT layer profiling 才能解释具体原因。

### 9.3 最终部署判断

- **精度优先、方案简单**：四个模型都可选择 TRT FP16。
- **ViT-B/16**：优先 MLP→FP16 混合精度。
- **Swin-T**：优先全量 INT8 v2。
- **ResNet50**：当前优先 FP16；若允许约 1 个百分点下降，可考虑浅层混合。
- **MobileNetV3-L**：优先 FP16；混合 INT8 精度合格，但吞吐仍比 FP16 低 3.8%。

## 10. 后续优化建议

1. **补齐 CNN 自动 fallback 搜索**：复用类别均衡子集和 ModelOpt 通配规则，
   输出每个候选组的精度、量化器数量和最终选择。
2. **做 TensorRT layer profiling**：重点解释为何 ViT/Mobile 混合 engine
   反而比全量 INT8 吞吐更高，并检查 reformat、copy 和未融合 Q/DQ。
3. **ResNet50**：细分 stem、layer1、layer2，寻找比 `layer1+2` 更小的 FP16
   集合；同时比较 percentile、MSE 和 cross-layer equalization。
4. **MobileNetV3**：按 depthwise、SE、hard-swish 分组搜索；尝试 SmoothQuant
   或其他 channel equalization。若仍需回退 blocks 0–5，则停止追求全量 INT8。
5. **ViT**：尝试 SmoothQuant 缩小 MLP 的 FP16 范围，并跟踪 logit cosine。
6. **必要时 QAT**：仅在部署约束明确要求 INT8、而 PTQ/混合精度仍不达标时使用。

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
Swin-T 可直接使用全量 INT8 v2，吞吐是 FP16 的 1.33x；
ViT-B/16 通过 MLP→FP16 将精度损失压到 0.070 个百分点；
MobileNetV3 证明 depthwise+SE 结构不能盲目使用 per-tensor 全量 INT8；
量化策略必须结合模型架构和实测敏感层，而不是统一套用一个配置。
```

因此，四模型的修正后基准、精度问题定位和第一轮混合精度优化已经完成。后续
工作的重点不再是重复跑基准，而是补齐 CNN 自动搜索、做 layer profiling，并
验证更小粒度的 ResNet/MobileNet 混合精度方案。
