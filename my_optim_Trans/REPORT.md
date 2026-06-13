# Transformer 模型推理量化加速实验报告

> **2026-06-13 修正后复测**
>
> ViT-B/16 已使用修正后的代码完成复测：TensorRT wrapper 不再重复同步，
> fallback 搜索中的 FP32、默认 INT8 和候选配置共享同一个 2,000 张类别均衡
> 子集，并使用 ModelOpt 通配量化规则。第五节中的 ViT-B/16 数据和第六节为
> 当前正式结果；修正前数据仅保留为历史对照。

**汇报身份**：学生阶段性工作总结  
**实验日期**：2026-06-08（初始实验）；2026-06-13（修正后 ViT 复测）<br>
**实验环境**：NVIDIA GeForce RTX 3080 Ti · CUDA 12.8 · TensorRT 10.15.1 · PyTorch 2.10  
**实验模型**：`vit_base_patch16_224`（ViT-B/16）、`swin_tiny_patch4_window7_224`（Swin-T）  
**评测数据集**：ImageNet val 50,000 张，校准集 500 张

---

## 一、工作目标

本阶段在 `my_optim_Trans/` 目录中，针对两个主流 Transformer 分类模型，完整走通从 PyTorch checkpoint 到 TensorRT INT8 量化部署的全链路，并对每一条优化路径做了准确率和吞吐率的量化评估。

核心目标：

1. 建立可复用的 Transformer 模型量化基准测试框架
2. 验证 TRT FP16 对 Transformer 的加速效果
3. 通过 ModelOpt PTQ + Q/DQ ONNX 实现 TRT INT8，并研究准确率与速度的权衡
4. 定位 ViT INT8 logit 分布失真的根因，探索 sensitive-layer fallback
5. 对齐 TRT builder 配置（`--fp16 --int8` 等价路径），验证实际吞吐提升

整体技术路线：

```
PyTorch checkpoint (timm pretrained)
    │
    ├─▶ FP32 / Torch FP16 (autocast) baseline ────── 准确率 + 延迟基准
    │
    ├─▶ ONNX export (dynamic batch, opset 17)
    │       │
    │       ├─▶ TRT FP16 engine ─────────────────── 零精度损失快速加速路径
    │       │
    │       └─▶ ModelOpt INT8 PTQ (Q/DQ ONNX)
    │               │
    │               ├─▶ TRT INT8 v1 (STRONGLY_TYPED only)
    │               └─▶ TRT INT8 v2 (WeaklyTyped + FP16 + INT8)  ← 最终优化版
    │
    └─▶ Sensitive-layer fallback 分析（ViT MLP 根因定位）
```

---

## 二、实验框架

`my_optim_Trans/` 包含 8 个阶段脚本，按依赖顺序独立运行，支持通过 `OPTIM_MODEL` 环境变量切换模型：

| 脚本 | 功能 |
|---|---|
| `common.py` | 模型构建、DataLoader、准确率评估、延迟测量共享工具 |
| `run_baseline.py` | FP32 + Torch FP16 全量评估 + 延迟 |
| `export_onnx_fp32.py` | 动态 batch ONNX 导出 + TRT 预解析验证 |
| `build_trt_engine.py` | TRT 引擎构建（FP16 / INT8，支持 opt level 配置） |
| `run_trt_eval.py` | TRT 引擎准确率 + 延迟评测 |
| `run_ptq_int8.py` | ModelOpt INT8 PTQ 校准 + Q/DQ ONNX 导出 |
| `run_sensitive_fallback.py` | 逐层 FP16 fallback 分析（可选） |
| `run_summary.py` | 汇总所有结果为 Markdown 表格 + JSON |

---

## 三、量化方案设计

### PTQ 校准策略

使用 NVIDIA ModelOpt 的 `INT8_DEFAULT_CFG`：

- 校准方法：MaxCalibrator（per-tensor activation，axis=0 per-channel weight）
- 校准集：500 张 ImageNet val 图像，16 张/批，共 31 批
- 量化目标层：所有 `Linear`（含 QKV、attn.proj、mlp.fc1/fc2、head）和 `Conv2d`（patch_embed）
- 排除层（默认不量化）：LayerNorm、Softmax、残差加法、GELU

ViT-B/16 共插入 **150 个量化器**（75 层 × input + weight），Swin-T 共 **159 个**。

### ONNX 导出

导出格式为带 `QuantizeLinear` / `DequantizeLinear`（Q/DQ）节点的 ONNX（opset 17），scale 常量内嵌于图中。TRT 通过读取这些节点决定哪些层使用 INT8 kernel。

---

## 四、TRT Builder 配置演进

这是本阶段的核心工程发现之一。

### v1 配置（初版，存在问题）

```python
network_flags = STRONGLY_TYPED   # 保证 Q/DQ 节点被识别
# 无 FP16/INT8 flag
config.builder_optimization_level = 3  # 默认
config.avg_timing_iterations = 1       # 默认
profile: opt_bs = 16, max_bs = 32
```

**问题**：`STRONGLY_TYPED` 模式下，未被 Q/DQ 覆盖的算子（LayerNorm、Softmax、GELU、残差加法等）默认回落到 **FP32**，导致这些算子成为吞吐瓶颈，INT8 的速度优势大打折扣。

### v2 配置（对齐 trtexec --fp16 --int8）

```python
network_flags = 0  # WeaklyTyped
config.set_flag(BuilderFlag.FP16)   # 非量化层回落 FP16（而非 FP32）
config.set_flag(BuilderFlag.INT8)   # Q/DQ 层使用 INT8 tensor-core kernel
config.builder_optimization_level = 5  # 穷举 kernel 搜索
config.avg_timing_iterations = 8       # 稳定 kernel timing
profile: opt_bs = 32, max_bs = 64      # 对齐实测 batch size
```

**TRT 10.15 实测约束**：`STRONGLY_TYPED` 网络在调用 `buildSerializedNetwork` 时**硬拒绝** FP16/INT8 flag（API error code 3），即使 `set_flag` 不报错。因此 "STRONGLY_TYPED + FP16 + INT8" 路径在 TRT 10.15 上不可用，必须使用 WeaklyTyped 网络。

---

## 五、实验结果

### 5.1 ViT-B/16（vit_base_patch16_224）

**FP32 基准**：top1 = 85.104%，bs=32 吞吐 = 356 img/s

| 精度模式 | top1% | Δtop1（百分点） | bs=1 延迟 | bs=32 吞吐 | bs=1 加速 | 吞吐加速 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 (torch) | 85.104 | — | 4.50 ms | 356 img/s | 1.00× | 1.00× |
| TRT FP16 | 85.104 | 0.000 | 2.95 ms | 2409 img/s | 1.53× | 6.77× |
| TRT INT8（全量） | 83.312 | −1.792 | **1.80 ms** | 1979 img/s | **2.50×** | 5.56× |
| **TRT INT8 混合精度（MLP→FP16）** | **85.034** | **−0.070** | **1.87 ms** | **2367 img/s** | **2.41×** | **6.65×** |

> `Δtop1 = 当前 top1 − FP32 top1`，负数表示精度下降。例如 −0.070
> 表示下降 0.070 个百分点，不是相对下降 0.070%。

**结论**：

- 全量 INT8 的 bs=1 延迟最低，但损失 1.792 个百分点，精度代价较大。
- MLP 回退 FP16 后，top1 比全量 INT8 恢复 **1.722 个百分点**，而 bs=1
  延迟只增加 **0.07 ms**（约 3.9%）。
- 混合精度吞吐达到 2367 img/s，是 TRT FP16 的 **98.3%**，同时比全量
  INT8 高 **19.6%**。这说明“位宽更低”不一定等于整网吞吐更高；在未做
  layer profiling 前，只能推测混合 engine 获得了更合适的 tactic/fusion。
- 最终推荐方案是 **MLP→FP16 的 INT8 混合精度**：相对 FP32 几乎无损，
  bs=1 仍加速约 **2.4×**。

<details>
<summary>修正前历史结果（不可与当前延迟直接比较）</summary>

旧脚本记录过 TRT FP16 3.49 ms / 1909 img/s、strongly-typed INT8
83.312% / 2.10 ms / 1504 img/s，以及旧 FP16+INT8 v2
83.376% / 1.54 ms / 3316 img/s。由于旧 wrapper 在
`execute_async_v3` 后重复同步，且 fallback 抽样口径不一致，这些数字仅用于
记录实验演进，不再作为最终结论。

</details>

### 5.2 Swin-T（swin_tiny_patch4_window7_224）

**FP32 基准**：top1 = 81.378%，bs=32 吞吐 = 695 img/s

| 精度 | top1% | Δtop1 | bs=1 延迟 | bs=32 吞吐 | vs TRT FP16 |
|---|---|---|---|---|---|
| FP32 (torch) | 81.378 | — | 6.63 ms | 695 img/s | — |
| Torch FP16 (autocast) | 81.380 | −0.002% | 8.76 ms | 1426 img/s | — |
| **TRT FP16** | **81.358** | **+0.020%** | **1.35 ms** | **4021 img/s** | 1.0× |
| TRT INT8 v1 (STRONGLY_TYPED) | 80.902 | +0.476% | 1.11 ms | 3081 img/s | 0.77× ↓ |
| **TRT INT8 v2 (FP16+INT8, opt5)** | **81.006** | **+0.372%** | **1.19 ms** | **5284 img/s** | **1.31× ↑** |

> INT8 v2 比 TRT FP16 快 **1.31×**，相比 FP32 快 **7.6×**。准确率损失 0.37%，几乎无损。

---

## 六、ViT INT8 Sensitive-Layer Fallback 分析

### 问题发现

ViT INT8 v1 的 logit cosine 相似度仅为 **0.790**，远低于 FP16 的 0.9999，说明即使 top1 只降了 1.8%，输出分布已发生明显变形。这对需要 softmax 置信度（如集成学习、知识蒸馏）的下游任务有影响。

### 根因定位

修正后的搜索让所有配置使用相同的 2,000 张类别均衡子集，结果如下：

| 配置 | 子集 top1% | 相对 FP32 下降 |
|---|---:|---:|
| FP32 | 84.7 | — |
| 默认 INT8 | 83.2 | 1.5 个百分点 |
| **全部 MLP (fc1+fc2) → FP16** | **84.4** | **0.3 个百分点** |

候选配置由该均衡子集筛选，随后重新导出 ONNX、构建 TensorRT engine，并在
ImageNet val 50,000 张图像上做最终验证：

| 配置 | 全量 top1% | 相对 FP32 下降 | bs=1 延迟 | bs=32 吞吐 |
|---|---:|---:|---:|---:|
| FP32 | 85.104 | — | 4.50 ms | 356 img/s |
| 全量 INT8 | 83.312 | 1.792 个百分点 | 1.80 ms | 1979 img/s |
| **MLP→FP16 混合精度** | **85.034** | **0.070 个百分点** | **1.87 ms** | **2367 img/s** |

### 关键结论

类别均衡搜索和全量 engine 评测得到一致结果，因此现在可以确认：
**MLP 层（fc1/fc2）是本次 ViT-B/16 PTQ 中 INT8 精度损失的主因。**

可能机制（与早期分布统计一致）：ViT MLP 中间层（GELU 后）的 activation
分布很宽，per-tensor 校准 amax 最高达 64–199，而 attn 层 amax 普遍在
2–20。使用单一 activation scale 时，少数离群值会放大量化步长，使大量普通
数值的表示变粗或在范围边界被截断；这一解释仍可通过 layer profiling 和逐层
误差统计进一步验证。

- 将全部 12 个 block 的 MLP 回退 FP16 后，全量 top1 从 83.312% 恢复到
  **85.034%**，只比 FP32 低 0.070 个百分点。
- 混合精度仅比全量 INT8 增加 0.07 ms 的 bs=1 延迟，同时 batch 吞吐更高，
  因而在本次实验中形成了明显更好的精度/速度折中。
- 后续仍可尝试 SmoothQuant，将激活离群值的量化难度迁移到权重，目标是减少
  MLP 的 FP16 比例，而不是再次证明 MLP 是否敏感。

---

## 七、工程经验总结

### 7.1 TRT 10.15 关键约束

| 约束 | 说明 |
|---|---|
| `STRONGLY_TYPED + FP16/INT8 flag` | **不兼容**，buildSerializedNetwork 返回 None（API error 3） |
| Q/DQ ONNX 的正确 build 路径 | `WeaklyTyped + FP16 + INT8`（等价于 trtexec --fp16 --int8） |
| `builder_optimization_level` | 默认 3，推荐 5（穷举 kernel 搜索，构建时间 +1–2 min，值得） |
| `avg_timing_iterations` | 默认 1，推荐 8（避免 kernel 选择不稳定） |
| `opt_bs` 对齐实测 batch | opt=32 vs opt=16 对 bs=32 的 kernel 选择影响显著 |

### 7.2 Transformer vs CNN 量化差异

| 特性 | ViT-B | Swin-T | CNN (ResNet 等) |
|---|---|---|---|
| 全量 INT8 精度损失 | 较大（MLP 分布宽） | 小（窗口注意力更规则） | 需按模型实测 |
| 主要敏感层 | MLP fc1/fc2（已复测确认） | 相对均匀 | 需做层敏感度搜索 |
| 推荐 INT8 路径 | MLP→FP16 混合精度 | 全量 INT8 v2 | 复测新版 builder 后决定 |
| per-tensor activation 适配性 | 较差（GELU 输出方差大） | 较好 | 好 |

### 7.3 各路径适用场景建议

| 路径 | 推荐场景 |
|---|---|
| TRT FP16 | 精度优先，零损失，对显存无强约束 → **默认首选** |
| 全量 TRT INT8 | bs=1 延迟优先，且可接受约 1.8 个百分点精度损失 |
| **MLP FP16 混合精度** | ViT 当前推荐方案：近乎无损，且保持 FP16 级 batch 吞吐 |

---

## 八、下一步方向

1. **SmoothQuant**：对 ViT MLP 的宽激活分布问题，将激活通道离群值的量化
   难度迁移到权重；部署时通常仍采用 per-tensor activation 与 per-channel
   weight。实际 cosine 改善需要重新实验，不能预设目标值。
2. **校准集优化**：当前 500 张校准图像全为 ImageNet，对 domain-specific 任务建议用目标域数据校准。
3. **FP8**：TRT 10.15 已支持 `BuilderFlag.FP8`，在 Ada 架构（RTX 40xx）上 FP8 速度接近 INT8 但精度更接近 FP16，值得试验。
4. **tiling_optimization_level**：当前 `TilingOptimizationLevel.NONE`，对 Swin 的窗口划分模式可能有进一步提升空间。
