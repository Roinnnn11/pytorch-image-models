# ViT INT8 固定 Batch 优化实验

本实验只针对 `vit_base_patch16_224`。默认约束是相对 PyTorch FP32 的
ImageNet Top-1 下降不超过 2.0 个百分点。

## 服务器准备

需要原实验环境中的 PyTorch、timm、NVIDIA ModelOpt、ONNX、TensorRT
10.15、CUDA 和 ImageNet validation 数据。先确认以下基准文件存在：

```text
my_optim_Trans/results/vit_base_patch16_224/baseline.json
my_optim_Trans/data/val/
```

若缺少 baseline，先运行：

```bash
cd my_optim_Trans
OPTIM_MODEL=vit_base_patch16_224 python run_baseline.py
```

## 查看将要执行的命令

```bash
cd my_optim_Trans
bash run_vit_int8_matrix.sh --dry-run
```

Dry-run 不加载 ModelOpt、TensorRT 或 ImageNet。

## 完整实验

```bash
cd my_optim_Trans
PY=/data1/liurongying/miniconda3/envs/deepburst/bin/python \
  bash run_vit_int8_matrix.sh \
  --batches 1,4,16,32 \
  --calib-samples 3000 \
  --max-top1-drop 2.0
```

默认候选包括 Max、MSE、SmoothQuant alpha=0.3/0.5/0.7/0.9。若已有
`run_scale_search.py` 和 `run_sensitive_fallback.py` 生成的 ONNX，也会自动加入
`scale_search` 和 `mlp_fallback`；缺失时只跳过这两个可选候选。

每个 Q/DQ ONNX 只校准一次，然后复用于四个固定引擎。每个引擎使用
`MIN=OPT=MAX=batch_size`，避免动态 profile 禁用固定形状 tactic。

完整矩阵最多构建 32 个 engine，优化等级 5 会消耗较长时间和磁盘空间。先做
小范围实验可用：

```bash
bash run_vit_int8_matrix.sh \
  --candidates max,mse,smoothquant_a0p5,mlp_fallback \
  --batches 1,4,16,32
```

已有成功 ONNX、engine 和评测 JSON 会自动复用。重新生成使用 `--force`。

## CUDA Graph

固定 batch 可以额外测试 CUDA Graph：

```bash
bash run_vit_int8_matrix.sh \
  --candidates max,mlp_fallback \
  --use-cuda-graph
```

结果 JSON 会记录 `cuda_graph_enabled`。捕获失败时脚本自动回退普通
`execute_async_v3`，并保存失败原因。

## 逐层 TensorRT Profile

Profiler 必须独立于正式性能测试运行，例如：

```bash
python run_trt_eval.py \
  --precision int8 \
  --engine-suffix int8_max_bs16.engine \
  --result-tag profile_int8_max_bs16 \
  --fixed-bs 16 \
  --max-batches 10 \
  --profile-layers
```

逐层数据写入结果 JSON 的 `layer_profile`，用于检查 Q/DQ reformat、未融合层和
慢 INT8 tactic。不要把 profile 运行的延迟作为正式吞吐数字。

## 结果

```text
results/vit_base_patch16_224/vit_int8_matrix.json
results/vit_base_patch16_224/vit_int8_matrix.md
results/vit_base_patch16_224/consistency_<candidate>.json
```

每个 batch 单独计算 Pareto 前沿。表中同时保存：

- 完整 ImageNet Top-1 / Top-5；
- CUDA event 测得的 GPU latency 和 throughput；
- 包含 DataLoader、H2D、同步和准确率统计的 E2E throughput；
- 实际评测图片数，确保固定 bs=32 的最后 16 张图片经过 padding 而不是丢弃。
- 同一 Q/DQ 候选的各 batch 引擎相对 bs=1 的 logit 最大误差和余弦相似度。

不要把 GPU throughput 与 E2E throughput 混为一个指标：前者定位 engine，后者
反映实际数据管线。不同 GPU、功耗、TensorRT 版本的绝对数值不能直接横向比较。
