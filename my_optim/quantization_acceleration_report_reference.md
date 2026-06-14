# pytorch-image-models 量化加速汇报参考版

> **2026-06-15 更新**：四个模型均已使用修正后的评测流程完成汇总。本稿中的
> 正式性能统一使用 bs=1 延迟和 bs=32 吞吐，精度差值使用百分点。

## 1. 汇报标题

基于 TensorRT 和 ModelOpt 的 timm 图像分类模型量化加速实验

## 2. 汇报主线

这次汇报的重点不是单纯展示结果表，而是说明我如何基于现有代码完成一条端到端量化加速 pipeline，并在 INT8 结果不符合预期时，通过代码和产物一步步定位问题。

整体思路是：

```text
统一模型和预处理
  -> 建立 PyTorch / TensorRT FP16 baseline
  -> 接入 ModelOpt PTQ Q/DQ INT8
  -> 发现 INT8 v1 问题
  -> 验证 Q/DQ 是否真的存在
  -> 对比 /optim 的 Q/DQ build 方式
  -> 补做 FP16+INT8 的 INT8 v2
  -> 再做校准和敏感层 fallback 分析
```

本次实验覆盖了 CNN 和 Transformer 两类模型：

- ResNet50
- MobileNetV3-Large
- ViT-B/16
- Swin-Tiny

实验平台为 RTX 3080 Ti，数据集为 ImageNet validation 50k。

## 3. 用到的关键技术

### 3.1 统一模型入口和预处理

代码上通过 `OPTIM_MODEL` 切换模型，并用 `timm.create_model` 创建模型。为了保证不同模型比较公平，预处理不是手写固定参数，而是通过 `resolve_data_config` 自动读取每个模型的 input size、mean、std、crop pct 和 interpolation。

这一步解决的是公平评估问题：不同模型输入尺寸和预处理不同，如果不统一，后面的精度对比没有意义。

### 3.2 ONNX 导出

使用 `torch.onnx.export` 导出 ONNX。这里不仅导出普通 FP32 ONNX，还额外导出 dynamic batch ONNX，用于 TensorRT optimization profile。

一个关键细节是，部分 exporter 会把 batch 固定住，所以我们使用 `dynamic_axes` 显式保留 batch 维度，避免 TensorRT engine 实际只能处理固定 batch。

### 3.3 TensorRT FP16

FP16 路线是：

```text
FP32 dynamic-batch ONNX -> TensorRT FP16 engine
```

技术点包括：

- TensorRT Python API
- `BuilderFlag.FP16`
- optimization profile
- serialized engine
- CUDA event latency 测量

TensorRT FP16 是后续 INT8 的强 baseline。

### 3.4 ModelOpt PTQ Q/DQ INT8

INT8 路线是本次重点：

```text
PyTorch model
  -> ModelOpt PTQ
  -> Q/DQ ONNX
  -> TensorRT INT8 engine
```

使用的核心 API 是：

```python
mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibration_loop)
```

`INT8_DEFAULT_CFG` 的主要含义：

- weight：INT8 per-channel
- activation：INT8 per-tensor
- calibration：默认 max

导出的 ONNX 里会显式包含：

- `QuantizeLinear`
- `DequantizeLinear`

所以这次 INT8 是 Q/DQ 量化路线，而不是普通 TensorRT calibrator。

### 3.5 校准策略探索

默认 max calibration 对部分模型不够好，所以额外探索了：

- percentile calibration
- mse calibration
- smoothquant

MobileNetV3 的精度下降比较明显，所以它是校准策略探索的重点模型。

### 3.6 TensorRT INT8 build 策略探索

这是本次最关键的探索点。

最初 INT8 v1 使用 strongly typed Q/DQ build。这个版本能跑通，但速度不理想。后来通过对比 `/data1/liurongying/workplace/optim` 发现，参考项目的 Q/DQ INT8 build 更接近：

```text
--fp16 --int8
```

这意味着：

```text
INT8 quantized layers + FP16 fallback for non-INT8 layers
```

所以我们补做了 INT8 v2：

- weakly typed network
- FP16 flag
- INT8 flag
- 更高 builder optimization level
- timing cache

v2 之后，ViT 和 Swin 的速度明显提升，说明 INT8 性能不只取决于 Q/DQ ONNX，也强依赖 TensorRT build 策略。

### 3.7 敏感层 FP16 fallback

为了分析 INT8 精度下降原因，还做了敏感层回退实验。做法是通过 ModelOpt quant config override，把部分层的 input quantizer 和 weight quantizer 关掉，让这些层保留 FP16。

尝试过的层包括：

- attention qkv
- attention projection
- classifier head
- MLP fc1 / fc2

ViT 的实验说明 MLP 层对 INT8 非常敏感。MLP 回退 FP16 可以明显恢复精度，但速度会下降。

## 4. 探索流程

整个探索过程可以按下面几步讲：

1. 先跑 PyTorch FP32 和 Torch FP16。
   - 目的：建立原始精度和速度基准。

2. 再做 TensorRT FP16。
   - 目的：建立强部署 baseline。
   - 结果：FP16 稳定、精度基本不变。

3. 接入 ModelOpt PTQ，导出 Q/DQ ONNX。
   - 目的：验证 INT8 pipeline。
   - 检查点：确认 ONNX 中有 Q/DQ 节点。

4. 构建 strongly typed INT8 v1。
   - 发现：有些模型速度没有超过 FP16，说明只跑通 Q/DQ 还不够。

5. 对比 `/optim` 的 Q/DQ 实现。
   - 发现：参考项目使用的是 `--fp16 --int8` 风格。
   - 推断：v1 可能缺少 FP16 fallback，导致非量化层拖慢。

6. 补做 INT8 v2。
   - 改成 `FP16 + INT8` build。
   - 四模型使用统一新版 builder 复测，Swin v2 吞吐是旧 v1 的 2.12x。

7. 针对精度问题做校准和敏感层分析。
   - ViT：MLP / attention / head fallback。
   - ResNet50：浅层 `layer1+2` fallback。
   - MobileNetV3：校准消融和 blocks 0–5 fallback。

这个流程体现的是：不是只跑一个脚本出结果，而是围绕异常现象做验证、对比和修正。

## 5. 结果只作为验证

结果不是汇报重点，但可以用来说明技术路线是否有效。

### 最终推荐结果

| 模型 | 推荐路径 | Top1 | ΔTop1 | bs1 延迟 | bs32 吞吐 |
|---|---|---:|---:|---:|---:|
| ResNet50 | TRT FP16 | 80.406 | +0.030 | 1.00 ms | 7831 img/s |
| MobileNetV3-Large | TRT FP16 | 75.762 | +0.006 | 0.639 ms | 20602 img/s |
| ViT-B/16 | INT8 混合（MLP→FP16） | 85.034 | −0.070 | 1.87 ms | 2367 img/s |
| Swin-Tiny | 全量 INT8 v2 | 81.006 | −0.372 | 1.25 ms | 5358 img/s |

## 6. 可以概括的技术结论

1. TensorRT FP16 是稳定部署 baseline。

2. Q/DQ ONNX 是正确的 INT8 表达方式。

3. INT8 性能不只取决于 Q/DQ 是否存在，还取决于 TensorRT build 策略。

4. `FP16 + INT8` build 对含有大量非量化算子的模型很重要。

5. 不同模型的 INT8 难点不同：
   - MobileNetV3：depthwise、SE 和 hard-swish 附近 activation 敏感。
   - ViT：MLP 层对量化敏感。
   - ResNet50：本次实验中浅层 `layer1+2` 更敏感。
   - Swin：INT8 v2 可以直接使用。

6. 混合精度 fallback 可以恢复精度，但不一定牺牲吞吐；最终性能还由 TensorRT
   tactic、fusion 和 reformat 决定。

## 7. 原始对比路径

整个推理加速流程分为四条路径：

1. PyTorch FP32
   - 作为精度和速度基线。

2. Torch FP16
   - 使用 PyTorch autocast。
   - 用来观察单纯半精度推理的收益。

3. TensorRT FP16
   - 先导出 FP32 ONNX。
   - 再构建 TensorRT FP16 engine。
   - 这是当前最稳定的加速方案。

4. TensorRT INT8
   - 使用 NVIDIA ModelOpt 做 PTQ。
   - 导出带 QuantizeLinear / DequantizeLinear 的 Q/DQ ONNX。
   - 再由 TensorRT 构建 INT8 engine。

可以概括为：

```text
PyTorch model -> ONNX -> TensorRT FP16
PyTorch model -> ModelOpt PTQ -> Q/DQ ONNX -> TensorRT INT8
```

## 8. Q/DQ 量化说明

这次 INT8 用的是 Q/DQ 方法，不是简单的 TensorRT calibrator。

Q/DQ 的含义是，在 ONNX 图中显式插入：

- `QuantizeLinear`
- `DequantizeLinear`

这样 TensorRT 可以根据图中的量化节点和 scale 信息选择 INT8 kernel。

本次检查了导出的 ONNX：

| 模型 | Q/DQ 节点情况 |
|---|---|
| ResNet50 | 有 Q/DQ |
| MobileNetV3 | 有 Q/DQ |
| ViT-B/16 | 有 Q/DQ |
| Swin-Tiny | 有 Q/DQ |

因此当前 INT8 路线本身是正确的 Q/DQ 量化路线。

## 9. 实验结果总表（备用）

| 模型 | 路径 | Top1 | ΔTop1 | bs1 延迟 | bs32 吞吐 |
|---|---|---:|---:|---:|---:|
| ResNet50 | FP32 | 80.376 | — | 4.31 ms | 1455 img/s |
| ResNet50 | TRT FP16 | 80.406 | +0.030 | 1.00 ms | 7831 img/s |
| ResNet50 | INT8 全量 | 78.738 | −1.638 | 0.888 ms | 10177 img/s |
| ResNet50 | INT8 混合 | 79.316 | −1.060 | 0.917 ms | 9461 img/s |
| MobileNetV3-L | FP32 | 75.756 | — | 4.68 ms | 4267 img/s |
| MobileNetV3-L | TRT FP16 | 75.762 | +0.006 | 0.639 ms | 20602 img/s |
| MobileNetV3-L | INT8 全量 | 59.516 | −16.240 | 0.916 ms | 13181 img/s |
| MobileNetV3-L | INT8 混合 | 75.658 | −0.098 | 0.662 ms | 19812 img/s |
| ViT-B/16 | FP32 | 85.104 | — | 4.50 ms | 356 img/s |
| ViT-B/16 | TRT FP16 | 85.104 | 0.000 | 2.95 ms | 2409 img/s |
| ViT-B/16 | INT8 全量 | 83.312 | −1.792 | 1.80 ms | 1979 img/s |
| ViT-B/16 | INT8 混合 | 85.034 | −0.070 | 1.87 ms | 2367 img/s |
| Swin-Tiny | FP32 | 81.378 | — | 6.63 ms | 695 img/s |
| Swin-Tiny | TRT FP16 | 81.358 | −0.020 | 1.35 ms | 4021 img/s |
| Swin-Tiny | INT8 v1 | 80.902 | −0.476 | 1.29 ms | 2523 img/s |
| Swin-Tiny | INT8 v2 | 81.006 | −0.372 | 1.25 ms | 5358 img/s |

结论：FP16 在四个模型上都稳定；INT8 是否值得使用则必须按架构判断。

## 10. 分模型结论

### ResNet50

ResNet50 的全量 INT8 吞吐达到 10177 img/s，但 Top1 下降 1.638 个百分点。
回退浅层 `layer1+2` 后精度恢复 0.578 个百分点，但仍低于 FP32 1.060 个
百分点。

结论：当前推荐 TRT FP16；如果部署能接受约 1 个百分点下降，再考虑浅层混合。

### MobileNetV3-Large

MobileNetV3 的全量 INT8 下降 16.240 个百分点。将 blocks 0–5 回退 FP16 后，
精度恢复到只差 0.098 个百分点，吞吐也达到 19812 img/s。

结论：混合精度有效，但只量化尾部少量 block，工程收益有限，默认仍推荐 FP16。

### ViT-B/16

ViT 全量 INT8 的 Top1 下降 1.792 个百分点。将全部 MLP 回退 FP16 后，Top1
达到 85.034%，只比 FP32 低 0.070 个百分点；bs1 仍为 1.87 ms，吞吐为
2367 img/s。

结论：ViT 当前推荐 MLP→FP16 混合精度。

### Swin-Tiny

Swin-Tiny 是当前全量 INT8 表现最健康的模型。v2 Top1 只下降 0.372 个
百分点，bs32 吞吐达到 5358 img/s，高于 TRT FP16 的 4021 img/s。

结论：Swin-Tiny 的 INT8 v2 已经同时具备较好的速度和精度，是本次 INT8 最成功的模型。

## 11. 为什么最初 INT8 v1 不理想

理论上，INT8 计算量更低，应该有更高加速潜力。但最初的 strongly typed INT8 v1 没有全面超过 TensorRT FP16，主要原因有几方面：

1. 对比对象是 TensorRT FP16，而不是普通 PyTorch FP16。

   TensorRT FP16 已经有 kernel fusion、Tensor Core、layout optimization，因此本身已经很快。

2. INT8 build 方式对结果影响很大。

   最初的 INT8 v1 使用 strongly typed Q/DQ 图，没有像参考项目 `/optim`
   那样走 `--fp16 --int8`。这会让没有被 Q/DQ 覆盖的算子可能以 FP32 执行，
   从而拖慢整体速度。后续改成 WeaklyTyped、FP16+INT8 和 opt level 5 后，
   Swin v2 的吞吐达到旧 v1 的 2.12 倍。

3. Transformer 中非量化算子较多。

   LayerNorm、Softmax、Transpose、Reshape 等算子不一定能 INT8 加速，但仍然占用推理时间。

4. PTQ 校准还比较基础。

   当前主要是 per-channel weight + per-tensor activation 的 PTQ。MobileNetV3 和 ViT 对 activation scale 比较敏感，需要更高级的校准或混合精度。

## 12. 本次工作的价值

本次工作不只是跑出了几组速度数据，更重要的是建立了完整的量化实验框架：

- 支持多模型切换。
- 支持统一预处理和统一评估。
- 支持 FP32、FP16、INT8 多精度对比。
- 支持 Q/DQ ONNX 导出。
- 支持 TensorRT engine 构建和全量 ImageNet 验证。
- 对 INT8 精度问题做了校准消融和敏感层回退分析。

最终得到的工程结论是：

```text
TensorRT FP16 是当前稳定可用的部署方案；
Swin-T 的全量 INT8 已同时满足速度和精度要求；
ViT-B/16 的 MLP 混合精度已把损失压到 0.070 个百分点；
ResNet50 和 MobileNetV3 仍需要更细粒度的 CNN fallback 搜索。
```

## 13. 后续工作

后续如果继续优化 INT8，可以从以下方向推进：

1. 补齐 CNN 自动 fallback 搜索。
   - 使用与 Transformer 相同的类别均衡子集。
   - 细分 ResNet stem/layer1/layer2。
   - 按 MobileNet depthwise/SE/hard-swish 分组。

2. 做 TensorRT layer profiling。
   - 检查哪些层是真 INT8。
   - 检查哪些层回退到 FP32 / FP16。
   - 检查是否有大量 reformat / copy。
   - 解释为什么部分混合 engine 比全量 INT8 更快。

3. 优化激活尺度。
   - ViT 尝试 SmoothQuant 缩小 MLP 回退范围。
   - ResNet 尝试 percentile/MSE 和 cross-layer equalization。
   - MobileNetV3 尝试 channel equalization；若仍需回退 blocks 0–5，则保留 FP16。

4. 必要时尝试 QAT。
   - 仅在部署明确要求 INT8、而 PTQ/混合精度仍不达标时使用。

## 14. 可直接汇报的话术

这次我完成了 timm 图像分类模型的端到端量化加速实验。我的重点不是单纯跑结果，而是搭建了一条从 PyTorch 到 ONNX，再到 TensorRT FP16 和 TensorRT INT8 的完整 pipeline。

技术上，FP16 部分主要使用 TensorRT Python API、dynamic batch ONNX 和 optimization profile。INT8 部分使用 NVIDIA ModelOpt 做 PTQ，导出带 QuantizeLinear 和 DequantizeLinear 的 Q/DQ ONNX，再交给 TensorRT 构建 engine。

探索过程中，最初 strongly typed INT8 v1 的速度不理想。我没有直接认为 INT8
无效，而是先检查 ONNX 里是否真的有 Q/DQ 节点，再对齐 `--fp16 --int8`
风格的 builder 配置。改成 WeaklyTyped、FP16+INT8、opt level 5 和 timing
cache 后，Swin v2 吞吐达到旧 v1 的 2.12 倍，证明 builder 配置会直接改变
INT8 的实际收益。

后续我又针对精度问题做了分组敏感层分析。ViT 的主要敏感区域是 MLP，
ResNet50 是浅层 layer1/2，MobileNetV3 是包含 depthwise 和 SE 的 blocks
0–5，而 Swin 不需要 fallback。这说明量化策略必须结合模型架构，不能给四个
模型套同一份 INT8 配置。

所以本次工作的结论是：TensorRT FP16 可以作为四模型的稳定 baseline；
Swin 可直接部署全量 INT8，ViT 适合 MLP 混合精度，ResNet 和 MobileNet
还需要更细粒度搜索。后续重点是 CNN 自动 fallback 和 TensorRT layer
profiling，而不是继续重复相同基准。
