# A100 AI Infra

这是一个面向NVIDIA A100/SM80的AI Infra学习与性能优化项目。建立可复现的PyTorch GEMM基线，后续将使用TileLang源码实现和优化BF16 GEMM、MoE Grouped GEMM与SwiGLU融合算子。

## 当前运行环境

| 项目 | 版本或状态 |
| --- | --- |
| 操作系统 | Linux |
| GPU | NVIDIA A100 80GB PCIe |
| GPU计算能力 | 8.0（SM80） |
| NVIDIA驱动 | 535.230.02 |
| CUDA Toolkit | 11.8 |
| C++编译器 | G++ 9.4.0 |
| Python | 3.10.21 |
| PyTorch | 2.3.1+cu118 |
| PyTorch编译CUDA版本 | 11.8 |
| TileLang | v0.1.9源码安装 |

更完整的环境说明见[docs/environment.md](docs/environment.md)。

## TileLang源码使用方式

TileLang源码将放在`kernel/tilelang`，并作为独立Git子模块管理，源码准备完成后，在项目根目录执行：

```zsh
# 激活项目使用的Conda环境
conda activate ai-infra

# 把项目内的TileLang源码以editable方式安装到当前环境
python -m pip install -e kernel/tilelang --no-build-isolation --no-deps
```

editable安装不会复制一份TileLang源码。修改`kernel/tilelang`中的源码后，当前Conda环境会直接使用修改后的代码。

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

CSV文件的列顺序固定为：

```text
M,N,K,dtype,latency_ms,tflops
```

## 目录结构

```text
ai-infra/
├── kernel/
│   ├── baseline/       PyTorch GEMM基线代码
│   ├── tilelang/       TileLang源码目录
│   ├── benchmark/      性能测试方案和记录
│   └── results/        CSV等实验结果
├── serving/
│   └── notes/          SGLang学习和实验记录
├── docs/               环境和项目文档
├── .gitignore
├── README.md
└── requirements.txt
```
