#!/usr/bin/env python3
# 使用当前Conda环境中的Python 3解释器执行本脚本。
"""在NVIDIA A100上运行一个最小、可复现的PyTorch GEMM性能基线。"""

from __future__ import annotations  # 延迟解析类型注解，减少类型名称前向引用带来的限制。

import argparse  # 解析命令行参数，例如矩阵尺寸、数据类型和迭代次数。
import csv  # 提供CSV表头和数据行的写入功能。
from pathlib import Path  # 以跨平台方式构造results目录和CSV文件路径。
from typing import Any  # 表示结果字典中的值可以是不同类型。

import torch  # 提供CUDA Tensor、矩阵乘法和CUDA Event计时功能。


def parse_args() -> argparse.Namespace:  # 定义并解析脚本支持的命令行参数。
    parser = argparse.ArgumentParser(description=__doc__)  # 创建参数解析器，并使用模块说明作为帮助信息。
    parser.add_argument("--m", type=int, default=1024)  # 设置矩阵A和结果C的行数M。
    parser.add_argument("--n", type=int, default=1024)  # 设置矩阵B和结果C的列数N。
    parser.add_argument("--k", type=int, default=1024)  # 设置矩阵A的列数以及矩阵B的行数K。
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")  # 选择FP16或BF16数据类型。
    parser.add_argument("--warmup", type=int, default=10)  # 设置正式计时前的预热次数。
    parser.add_argument("--iters", type=int, default=100)  # 设置正式计时阶段重复执行GEMM的次数。
    parser.add_argument("--device", default="cuda")  # 设置运行设备，默认使用当前可见的CUDA GPU。
    return parser.parse_args()  # 解析命令行并返回包含所有参数的Namespace对象。


def dtype_from_name(name: str) -> torch.dtype:  # 把命令行中的数据类型名称转换为PyTorch数据类型。
    return torch.float16 if name == "fp16" else torch.bfloat16  # fp16对应float16，其余合法选项对应bfloat16。


def save_result_to_csv(  # 定义把一次benchmark结果追加保存到CSV的函数。
    m: int,  # 接收矩阵维度M。
    n: int,  # 接收矩阵维度N。
    k: int,  # 接收矩阵维度K。
    dtype: str,  # 接收本次benchmark使用的数据类型名称。
    latency_ms: float,  # 接收单次GEMM的平均延迟，单位为毫秒。
    tflops: float,  # 接收本次GEMM的计算吞吐量。
) -> Path:  # 返回实际写入的CSV文件路径。
    results_dir = Path(__file__).resolve().parents[1] / "results"  # 定位kernel/results目录，不依赖运行脚本时的当前目录。
    results_dir.mkdir(parents=True, exist_ok=True)  # 确保results目录存在；目录已存在时不报错。
    csv_path = results_dir / "torch_gemm_results.csv"  # 设置统一的CSV结果文件名。
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0  # 文件不存在或为空时需要写入表头。
    fieldnames = ["M", "N", "K", "dtype", "latency_ms", "tflops"]  # 固定CSV列名及排列顺序。
    row = {  # 组织本次需要写入CSV的一行结果。
        "M": m,  # 保存矩阵维度M。
        "N": n,  # 保存矩阵维度N。
        "K": k,  # 保存矩阵维度K。
        "dtype": dtype,  # 保存输入数据类型。
        "latency_ms": round(latency_ms, 6),  # 保存单次GEMM平均延迟，并保留6位小数。
        "tflops": round(tflops, 6),  # 保存计算吞吐量，并保留6位小数。
    }  # 结束CSV数据行字典。
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:  # 以追加模式打开UTF-8格式的CSV，保留历史结果。
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)  # 创建按指定列顺序写入字典的CSV writer。
        if needs_header:  # 只在新文件或空文件中写入一次表头。
            writer.writeheader()  # 写入M、N、K、dtype、latency_ms和tflops列名。
        writer.writerow(row)  # 将当前benchmark结果追加为CSV的一行。
    return csv_path  # 返回CSV路径，供终端输出显示保存位置。


def main() -> None:  # 定义脚本的主要执行流程。
    args = parse_args()  # 读取用户传入的矩阵尺寸、数据类型和迭代次数。
    if not torch.cuda.is_available():  # 检查PyTorch是否能够发现并使用CUDA GPU。
        raise RuntimeError(  # CUDA不可用时立即停止，避免得到无意义的CPU benchmark。
            "CUDA不可用。请激活ai-infra的Conda环境。"  # 提示用户首先激活正确的Conda环境。
        )  # 结束RuntimeError的构造。
    if min(args.m, args.n, args.k, args.warmup, args.iters) <= 0:  # 确保尺寸和循环次数全部为正数。
        raise ValueError("矩阵尺寸、预热次数和正式迭代次数必须为正数。")  # 参数非法时给出明确错误。

    device = torch.device(args.device)  # 将设备字符串转换为PyTorch设备对象。
    dtype = dtype_from_name(args.dtype)  # 将fp16或bf16字符串转换为实际Tensor数据类型。
    a = torch.randn((args.m, args.k), device=device, dtype=dtype)  # 在GPU上创建形状为[M,K]的随机矩阵A。
    b = torch.randn((args.k, args.n), device=device, dtype=dtype)  # 在GPU上创建形状为[K,N]的随机矩阵B。

    for _ in range(args.warmup):  # 重复执行若干次GEMM，使CUDA上下文、缓存和库进入稳定状态。
        _ = a @ b  # 执行A×B，但丢弃预热阶段的结果，不把它计入正式性能数据。
    torch.cuda.synchronize(device)  # 等待所有预热kernel完成，因为CUDA默认是异步执行的。

    start = torch.cuda.Event(enable_timing=True)  # 创建用于记录计时起点的CUDA Event。
    end = torch.cuda.Event(enable_timing=True)  # 创建用于记录计时终点的CUDA Event。
    start.record()  # 将起点Event记录到当前CUDA stream中。
    for _ in range(args.iters):  # 按指定次数重复执行GEMM，以减小单次测量噪声。
        c = a @ b  # 执行矩阵乘法 [M,K]×[K,N]，得到形状为 [M,N] 的结果C。
    end.record()  # 在最后一次GEMM之后记录终点Event。
    end.synchronize()  # 等待终点Event完成，确保所有被测kernel已经执行完毕。

    total_ms = start.elapsed_time(end)  # 计算起点和终点之间的GPU总耗时，单位为毫秒。
    latency_ms = total_ms / args.iters  # 用总耗时除以迭代次数，得到单次GEMM的平均延迟。
    tflops = (2.0 * args.m * args.n * args.k) / (latency_ms * 1e9)  # 根据2MNK次浮点操作计算每秒万亿次运算。
    csv_path = save_result_to_csv(  # 将用户要求的六个字段追加保存到results目录下的CSV。
        m=args.m,  # 传入矩阵维度M。
        n=args.n,  # 传入矩阵维度N。
        k=args.k,  # 传入矩阵维度K。
        dtype=args.dtype,  # 传入数据类型名称。
        latency_ms=latency_ms,  # 传入未四舍五入的平均延迟。
        tflops=tflops,  # 传入未四舍五入的计算吞吐量。
    )  # 完成CSV追加写入并取得文件路径。
    project_root = Path(__file__).resolve().parents[2]  # 定位项目根目录，用于生成不含个人目录的相对路径。
    relative_csv_path = csv_path.relative_to(project_root)  # 将CSV绝对路径转换为项目内相对路径，避免输出主机个人信息。

    result: dict[str, Any] = {  # 组织本次benchmark的参数、环境和性能结果。
        "M": args.m,  # 使用大写字段M记录矩阵行数。
        "N": args.n,  # 使用大写字段N记录矩阵列数。
        "K": args.k,  # 使用大写字段K记录矩阵公共维度。
        "dtype": args.dtype,  # 记录输入矩阵的数据类型。
        "device": torch.cuda.get_device_name(device),  # 记录实际运行benchmark的GPU型号。
        "capability": list(torch.cuda.get_device_capability(device)),  # 记录GPU Compute Capability，例如A100为[8,0]。
        "torch": torch.__version__,  # 记录PyTorch版本，方便日后复现实验。
        "torch_cuda": torch.version.cuda,  # 记录当前PyTorch wheel编译时使用的CUDA版本。
        "latency_ms": round(latency_ms, 6),  # 记录单次GEMM的平均延迟，并保留6位小数。
        "tflops": round(tflops, 6),  # 记录计算吞吐量TFLOPS，并保留6位小数。
        "output_shape": list(c.shape),  # 记录结果矩阵C的形状，用于检查维度是否正确。
        "csv_path": relative_csv_path.as_posix(),  # 输出项目内CSV相对路径，不暴露主机用户名或个人目录。
    }  # 结束结果字典。
    for field, value in result.items():  # 按字典中的顺序逐个读取字段和对应数据。
        print(f"{field}: {value}")  # 使用“字段: 数据”的格式输出当前字段。


if __name__ == "__main__":  # 仅在直接运行该文件时执行main，作为模块导入时不会自动运行。
    main()  # 启动参数解析、GPU预热、正式计时和结果输出流程。
