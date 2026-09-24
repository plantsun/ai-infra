# 软件与硬件环境

本文档记录PyTorch GEMM和TileLang源码编译所使用的可复现环境。

## 硬件环境

- GPU：NVIDIA A100 80GB PCIe
- GPU计算能力：8.0（SM80）
- BF16支持：是
- NVIDIA驱动：535.230.02

## 系统与编译环境

- 操作系统：Linux
- CUDA Toolkit：11.8
- C++编译器：G++ 9.4.0

## Python环境

- Python：3.10.21
- PyTorch：2.3.1+cu118
- PyTorch编译CUDA版本：11.8
- NumPy：2.2.6
- Ninja：1.13.2
- pytest：9.1.1

## 使用pip重建基础环境

在项目根目录执行：

```zsh
# 创建Python 3.10的Conda环境。
conda create -n ai-infra python=3.10 -y

# 激活Conda环境
conda activate ai-infra

# 从requirements.txt安装PyTorch和构建依赖。
python -m pip install --no-cache-dir -r requirements.txt
```

## 配置CUDA 11.8

```zsh
# 指定CUDA 11.8根目录
export CUDA_HOME=/usr/local/cuda-11.8

# 优先使用CUDA 11.8的nvcc
export PATH=$CUDA_HOME/bin:$PATH

# 加载CUDA 11.8运行库
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## 验证环境

```zsh
# 输出Python版本，应为3.10.x
python --version

# 输出nvcc版本，应显示release 11.8
nvcc --version

# 输出PyTorch、CUDA和GPU信息
python -c 'import torch; print("PyTorch版本:", torch.__version__); print("CUDA版本:", torch.version.cuda); print("GPU型号:", torch.cuda.get_device_name(0)); print("计算能力:", torch.cuda.get_device_capability(0)); print("BF16支持:", torch.cuda.is_bf16_supported())'
```

结果应为：

```text
PyTorch版本: 2.3.1+cu118
CUDA版本: 11.8
GPU型号: NVIDIA A100 80GB PCIe
计算能力: (8, 0)
BF16支持: True
```
