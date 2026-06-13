# Transformer 模型推理量化加速实验报告

> **2026-06-13 代码审查修正**
>
> 下文性能数据是在旧评测脚本上得到的历史记录。当前脚本已经去除 TensorRT
> wrapper 内的重复同步，并将 fallback 搜索改为：FP32、默认 fake-quant 和
> 所有候选配置共享同一个类别均衡子集，同时使用 ModelOpt 通配量化规则。
> 因此第六节的 MLP 根因结论目前应视为待复现实验假设，需以新版
> `results/<model>/fallback_search.json` 和重新构建的 engine 为准。

**汇报身份**：学生阶段性工作总结  
**实验日期**：2026-06-08  
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

| 精度 | top1% | Δtop1 | bs=1 延迟 | bs=32 吞吐 | vs TRT FP16 |
|---|---|---|---|---|---|
| FP32 (torch) | 85.104 | — | 4.50 ms | 356 img/s | — |
| Torch FP16 (autocast) | 85.110 | −0.006% | 4.74 ms | 1087 img/s | — |
| **TRT FP16** | **85.104** | **+0.000%** | **3.49 ms** | **1909 img/s** | 1.0× |
| TRT INT8 v1 (STRONGLY_TYPED) | 83.312 | +1.792% | 2.10 ms | 1504 img/s | 0.79× ↓ |
| **TRT INT8 v2 (FP16+INT8, opt5)** | **83.376** | **+1.728%** | **1.54 ms** | **3316 img/s** | **1.74× ↑** |

> INT8 v2 比 TRT FP16 快 **1.74×**，相比 FP32 快 **9.3×**。准确率损失 1.73%。

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

通过穷举 4 类 fallback 配置（在 FakeQuant 阶段保留目标层为 FP16，对全量 50k val 重新评估），发现：

| 配置 | top1% | logit cosine | 活跃量化器 |
|---|---|---|---|
| INT8 全量（基准） | 83.312 | 0.790 | 150/150 |
| head → FP16 | 83.298 | — | 98/150 |
| attn.qkv + attn.proj → FP16 | 83.460 | 0.811 | 50/150 |
| **全部 MLP (fc1+fc2) → FP16** | **85.012** | **0.988** | **52/150** |
| 深层 MLP only (blocks 8-11) → FP16 | 83.280 | 0.782 | 84/150 |

### 关键结论

旧实验提示 **MLP 层（fc1/fc2）可能是 ViT INT8 精度损失的主因**，但新版
类别均衡搜索尚未在原 GPU 环境复跑，因此不能把它作为已经严格证明的结论。

原因：ViT MLP 中间层（GELU 后）的 activation 分布极宽，per-tensor 校准 amax 最高达 64–199，严重超出 INT8 表示范围（−128 到 127 × scale），导致大量数值被截断。而 attn 层 amax 普遍在 2–20 以内，per-tensor INT8 完全可承受。

- 将全部 12 个 block 的 MLP 回退 FP16 后，top1 **恢复至 85.01%**（仅比 FP32 差 0.09%），logit cosine **从 0.790 提升至 0.988**。
- 仅回退深层（blocks 8–11）反而使 cosine 降至 0.782，因为浅层 MLP 误差会被后续层部分吸收，局部回退打破了这种补偿。
- **待验证方向**：对 ViT 使用 INT8 时，可尝试 SmoothQuant 将激活离群值的
  量化难度迁移到权重，或把敏感 MLP 保留为 FP16，再比较精度和速度。

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
| INT8 精度损失 | 较大（MLP 分布宽） | 小（窗口注意力更规则） | 通常极小 |
| 主要敏感层 | MLP fc1/fc2 | 相对均匀 | 最后几层 Conv |
| TRT INT8 vs FP16 速度 | 1.74× | 1.31× | 通常 1.5–2.5× |
| per-tensor activation 适配性 | 较差（GELU 输出方差大） | 较好 | 好 |

### 7.3 各路径适用场景建议

| 路径 | 推荐场景 |
|---|---|
| TRT FP16 | 精度优先，零损失，对显存无强约束 → **默认首选** |
| TRT INT8 v2 | 吞吐优先，1–2% 精度可接受，线上高并发推理 |
| MLP FP16 混合精度 | 需要可信 logit 分布（蒸馏/集成）且允许一定速度代价 |

---

## 八、下一步方向

1. **SmoothQuant**：对 ViT MLP 的宽激活分布问题，将激活通道离群值的量化
   难度迁移到权重；部署时通常仍采用 per-tensor activation 与 per-channel
   weight。实际 cosine 改善需要重新实验，不能预设目标值。
2. **校准集优化**：当前 500 张校准图像全为 ImageNet，对 domain-specific 任务建议用目标域数据校准。
3. **FP8**：TRT 10.15 已支持 `BuilderFlag.FP8`，在 Ada 架构（RTX 40xx）上 FP8 速度接近 INT8 但精度更接近 FP16，值得试验。
4. **tiling_optimization_level**：当前 `TilingOptimizationLevel.NONE`，对 Swin 的窗口划分模式可能有进一步提升空间。
