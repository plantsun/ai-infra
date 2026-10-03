# A100 AI Infra

这是一个面向NVIDIA A100/SM80的AI Infra学习与性能优化项目。当前阶段已经建立PyTorch GEMM基线、TileLang源码开发环境，以及遵守“Correctness→Performance”顺序的统一GEMM基准。后续将继续实现和优化BF16 GEMM、MoE Grouped GEMM与SwiGLU融合算子。

## 当前运行环境

| 项目 | 版本或状态 |
| --- | --- |
| 操作系统 | Linux |
| GPU | NVIDIA A100 80GB PCIe |
| GPU计算能力 | 8.0（SM80） |
| NVIDIA驱动 | 535.230.02 |
| 系统CUDA Toolkit | 12.5 |
| CUDA主机编译器 | Conda G++12.4.0 |
| Python | 3.10.21 |
| PyTorch | 2.6.0+cu124 |
| PyTorch编译CUDA版本 | 12.4 |
| TileLang | main分支源码，`0.1.15+cuda.git994b44ec` |
| TileLang子模块提交 | `994b44ec` |

更完整的安装、环境变量和验证方法见[docs/environment.md](docs/environment.md)。

## 初始化TileLang源码子模块

克隆外层仓库后，在项目根目录执行：

```zsh
# 下载并检出外层仓库记录的TileLang源码提交。
git submodule update --init --recursive

# 激活项目使用的Conda环境。
conda activate ai-infra

# 以editable模式构建并安装项目内的TileLang源码。
python -m pip install -e kernel/tilelang -v --no-build-isolation --no-deps
```

editable安装不会复制另一份TileLang源码。Python直接从`kernel/tilelang`加载包；修改Python源码后通常无需重新安装，修改C++、CUDA或编译器源码后需要重新执行构建安装命令。

## 运行TileLang Quick Start

在项目根目录执行：

```zsh
# 激活已经构建TileLang源码的环境。
conda activate ai-infra

# 运行TileLang仓库自带的Quick Start示例。
python kernel/tilelang/examples/quickstart.py
```

该示例会完成以下工作：

1. 使用TileLang DSL定义矩阵乘法内核。
2. 为当前GPU编译内核。
3. 创建测试输入并运行内核。
4. 与PyTorch参考结果进行正确性比较。
5. 输出TileLang内核和参考实现的延迟或性能信息。

Quick Start是上游教学示例，其中包含ReLU后处理。公平比较纯GEMM时应使用本项目的`kernel/benchmark/bench_gemm.py`。

## 统一GEMM正确性与性能基准

基准固定测试：

```text
M: 128,256,512,1024,2048,4096
N: 4096
K: 4096
BM: 128
BN: 128
BK: 32
```

运行FP16基准：

```zsh
# 运行Torch与TileLang的FP16纯GEMM正确性和性能比较。
python kernel/benchmark/bench_gemm.py --dtype fp16
```

运行BF16基准：

```zsh
# 运行Torch与TileLang的BF16纯GEMM正确性和性能比较。
python kernel/benchmark/bench_gemm.py --dtype bf16
```

脚本对每一个shape严格执行：

```text
PyTorch参考结果
↓
TileLang输出
↓
torch.testing.assert_close
↓
correctness = PASS
↓
Torch与TileLang性能计时
```

任何shape验证失败时，脚本都会输出`correctness = FAIL`并立即停止，不会为该shape记录性能结果。FP16和BF16使用适合低精度计算的`rtol`与`atol`，不要求逐位相同。

终端输出格式如下：

```text
phase: Correctness
backend: tilelang
M: 128
N: 4096
K: 4096
dtype: fp16
correctness = PASS
phase: Performance
backend: torch
latency_ms: <实测值>
tflops: <实测值>
```

统一结果会覆盖写入`kernel/results/gemm_benchmark_results.csv`，列顺序固定为：

```text
backend,M,N,K,dtype,BM,BN,BK,latency_ms,tflops
```

> Torch行的`BM、BN、BK`为空，TileLang行会记录实际分块参数。

## PyTorch GEMM基线

运行命令：

```zsh
# 运行形状为[1024,1024]×[1024,1024]的FP16 GEMM基线
python kernel/baseline/torch_gemm.py --m 1024 --n 1024 --k 1024 --dtype fp16

# 运行同一矩阵形状的BF16 GEMM基线
python kernel/baseline/torch_gemm.py --m 1024 --n 1024 --k 1024 --dtype bf16
```

脚本会执行以下功能：

1. 在GPU上创建形状为`[M,K]`和`[K,N]`的随机矩阵。
2. 执行预热，避免CUDA初始化影响正式计时。
3. 使用CUDA Event测量多次PyTorch矩阵乘法的平均延迟。
4. 根据`2×M×N×K`计算TFLOPS。
5. 将`M,N,K,dtype,latency_ms,tflops`追加保存到`kernel/results/torch_gemm_results.csv`。
6. 使用“字段: 数据”格式输出参数、运行环境、性能结果和结果矩阵形状。

终端输出格式如下，延迟和TFLOPS会随机器状态变化：

```text
M: 1024
N: 1024
K: 1024
dtype: fp16
device: NVIDIA A100 80GB PCIe
capability: [8, 0]
torch: 2.3.1+cu118
torch_cuda: 11.8
latency_ms: <实测值>
tflops: <实测值>
output_shape: [1024, 1024]
csv_path: kernel/results/torch_gemm_results.csv
```

统一结果会覆盖写入`kernel/results/torch_gemm_results.csv`，列顺序固定为：

```text
M,N,K,dtype,latency_ms,tflops
```

## 目录结构

```text
ai-infra/
├── kernel/
│   ├── baseline/       PyTorch GEMM基线代码
│   ├── benchmark/      正确性与性能基准代码
│   ├── results/        CSV等实验结果
│   └── tilelang/       TileLang源码Git子模块
├── serving/
│   └── notes/          SGLang学习和实验记录
├── docs/               环境和项目文档
├── .gitignore
├── .gitmodules
├── README.md
└── requirements.txt
```
