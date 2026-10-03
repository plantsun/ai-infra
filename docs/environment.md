# 软件与硬件环境

本文档记录PyTorch GEMM、TileLang源码构建和A100性能测试使用的当前可复现环境。

## 硬件环境

- GPU：NVIDIA A100 80GB PCIe
- GPU计算能力：8.0（SM80）
- BF16支持：是
- NVIDIA驱动：535.230.02

## CUDA与编译环境

- 操作系统：Linux
- 系统CUDA Toolkit：12.5
- NVCC：CUDA12.5，build`cuda_12.5.r12.5/compiler.34177558_0`
- CUDA主机编译器：Conda G++12.4.0
- Conda编译器包：`gcc_linux-64=12.4.0`、`gxx_linux-64=12.4.0`

## Python环境

- Conda环境名称：`ai-infra`
- Python：3.10.21
- PyTorch：2.6.0+cu124
- PyTorch编译CUDA版本：12.4
- TileLang：`0.1.15+cuda.git994b44ec`
- TileLang源码提交：`994b44ec`
- NumPy：2.2.6
- Ninja：1.13.2
- Cython：3.1.8
- scikit-build-core：1.0.3
- CMake：4.4.3
- patchelf：0.19.1.0
- z3-solver：4.15.4.0
- apache-tvm-ffi：0.1.12
- pytest：9.1.1

## 当前Conda环境变量

激活conda环境后使用以下变量：

```text
CUDA_HOME: /usr/local/cuda-12.5
CUDACXX: /usr/local/cuda-12.5/bin/nvcc
CXX: $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
CUDAHOSTCXX: $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
NVCC_CCBIN: $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
LD_PRELOAD: /lib/x86_64-linux-gnu/libcuda.so.1
```

`LD_PRELOAD`确保运行源码构建版TileLang时优先加载真实NVIDIA驱动库，避免错误加载CUDA Toolkit中的stub库。

## 从头创建当前Conda环境

以下命令记录当前环境的重建方法。执行前应先确认系统已经安装CUDA Toolkit12.5，并且`/usr/local/cuda-12.5`存在。

```zsh
# 创建使用Python3.10的ai-infra环境。
conda create -n ai-infra python=3.10 -y

# 激活新环境。
conda activate ai-infra

# 在Conda环境内安装独立的GCC/G++12.4工具链。
conda install -c conda-forge gcc_linux-64=12.4 gxx_linux-64=12.4 -y

# 从requirements.txt安装PyTorch和构建依赖。
python -m pip install --no-cache-dir -r requirements.txt
```

Conda G++12.4不属于pip包，仍需使用上面的`conda install`命令单独安装；TileLang本体也仍需从`kernel/tilelang`源码子模块单独构建。

## 配置Conda环境变量

在已经激活conda环境后执行：

```zsh
# 持久化CUDA12.5工具链、Conda G++12和NVIDIA驱动库设置。
conda env config vars set CUDA_HOME=/usr/local/cuda-12.5 CUDACXX=/usr/local/cuda-12.5/bin/nvcc CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" NVCC_CCBIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
```

## 获取和构建TileLang源码

项目使用`kernel/tilelang`作为规范Git子模块。在外层仓库根目录执行：

```zsh
# 获取外层仓库固定的TileLang源码提交。
git submodule update --init --recursive

# 使用当前环境中的CUDA12.5和G++12.4构建editable版本。
python -m pip install -e kernel/tilelang -v --no-build-isolation --no-deps
```

不需要在每次运行Python脚本前重复构建。只有首次安装、切换TileLang提交、清理构建目录，或者修改C++、CUDA与编译器源码后，才需要重新运行构建命令。

## 验证环境

在项目根目录和已激活的conda环境中执行：

```zsh
# 输出Python版本，应为3.10.21。
python --version

# 输出系统CUDA编译器版本，应显示release12.5。
$CUDA_HOME/bin/nvcc --version

# 输出Conda C++编译器版本，应显示G++12.4.0。
$CXX --version

# 输出PyTorch、TileLang、CUDA和GPU信息。
python -c 'import sys,torch,tilelang; print("Python:",sys.version.split()[0]); print("PyTorch:",torch.__version__); print("PyTorch CUDA:",torch.version.cuda); print("TileLang:",tilelang.__version__); print("GPU:",torch.cuda.get_device_name(0)); print("Compute Capability:",torch.cuda.get_device_capability(0)); print("BF16:",torch.cuda.is_bf16_supported())'
```

当前环境的关键输出应为：

```text
Python: 3.10.21
PyTorch: 2.6.0+cu124
PyTorch CUDA: 12.4
TileLang: 0.1.15+cuda.git994b44ec
GPU: NVIDIA A100 80GB PCIe
Compute Capability: (8, 0)
BF16: True
```

## 运行验证

验证TileLang源码环境：

```zsh
# 编译并运行TileLang仓库自带的Quick Start示例。
python kernel/tilelang/examples/quickstart.py
```

验证本项目的正确性优先GEMM基准：

```zsh
# 对全部指定shape先验证FP16结果，再比较Torch和TileLang性能。
python kernel/benchmark/bench_gemm.py --dtype fp16
```

每一个shape都必须先出现以下输出，之后才应出现性能字段：

```text
correctness = PASS
```
