# INT8 Scale Search

这是一套新增实验，不会替换或修改原来的 ModelOpt PTQ、percentile/MSE
校准和 FP16 fallback 搜索。

## 搜索的是什么

ModelOpt 默认先为每个 activation quantizer 得到 `amax`。对称 INT8 中：

```text
scale = amax / 127
```

新方法对 MobileNetV3 和 ViT 的不同架构模块分别尝试：

```text
new_amax = ModelOpt_amax * ratio
ratio in [1.0, 0.9, 0.8, 0.7, 0.6]
```

`ratio` 变小会裁掉少量极端激活值，同时减小 INT8 量化步长。每个候选值都在
同一个类别均衡 ImageNet 子集上测试，以 Top1 作为主要目标，逐组保留最优值。
这属于手动定义搜索空间、程序自动遍历候选配置的方法。

搜索完成后，最佳 `_amax` 会留在 ModelOpt 的 `TensorQuantizer` 中。导出 ONNX
时，它们会成为 Q/DQ 节点的 scale 常量，因此 TensorRT 使用的就是搜索后的
scale，而不是默认 scale。

## MobileNetV3

搜索组：stem、depthwise、SE、pointwise、head。

```bash
cd pytorch-image-models
PY=/data1/liurongying/miniconda3/envs/deepburst/bin/python \
  bash my_optim/run_mobile_scale_search.sh
```

## ViT-B/16

搜索组：patch embedding、attention QKV、attention projection、MLP FC1、
MLP FC2、classification head。

```bash
cd pytorch-image-models
PY=/data1/liurongying/miniconda3/envs/deepburst/bin/python \
  bash my_optim_Trans/run_vit_scale_search.sh
```

## 缩短运行时间

默认用 1000 张类别均衡图片和 5 个倍率。时间特别紧时可以先跑：

```bash
SEARCH_SAMPLES=500 SCALE_RATIOS=1.0,0.8,0.6 \
  PY=/data1/liurongying/miniconda3/envs/deepburst/bin/python \
  bash my_optim/run_mobile_scale_search.sh
```

正式汇报建议至少使用默认的 1000 张配置。搜索结束后，一键脚本还会构建
TensorRT engine，并在完整 ImageNet validation set 上测 Top1、Top5、bs=1
延迟和 bs=32 吞吐。

## 独立输出

每个模型都会生成：

```text
results/<model>/scale_search.json
onnx/<model>/<model>_int8_qdq_scale_search_inline.onnx
engines/<model>/<model>_int8_scale_search.engine
results/<model>/trt_int8_scale_search.json
```

`scale_search.json` 保存每组所有候选倍率的精度、最终倍率，以及每个 activation
quantizer 的最终 `amax` 和 `scale`，可以直接用于汇报和复现实验。
