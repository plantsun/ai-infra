"""比较PyTorch与TileLang纯GEMM的正确性和性能。"""  # 说明本脚本只测试不带激活函数的矩阵乘法。

import argparse  # 导入命令行参数解析模块，允许从终端配置精度和计时次数。
import csv  # 导入CSV模块，用于把统一格式的性能结果写入文件。
from pathlib import Path  # 导入路径对象，用于构造不包含个人信息的项目相对结果路径。
from typing import Callable  # 导入可调用对象类型，用于标注统一计时函数的参数。

import torch  # 导入PyTorch，用于生成输入、计算参考结果和执行Torch基线。
import tilelang  # 导入TileLang，用于定义并编译GPU GEMM内核。
import tilelang.language as T  # 导入TileLang语言接口，并使用T作为简短别名。


M_LIST = [  # 定义需要测试的全部M维度。
    128,  # 测试较小M，模拟更接近小批量的矩阵乘法。
    256,  # 测试M等于256的矩阵乘法。
    512,  # 测试M等于512的矩阵乘法。
    1024,  # 测试M等于1024的矩阵乘法。
    2048,  # 测试M等于2048的矩阵乘法。
    4096,  # 测试M等于4096的方阵乘法。
]  # 结束M维度列表。
N = 4096  # 固定输出矩阵的列数N为4096。
K = 4096  # 固定归约维度K为4096。
BM = 128  # 设置TileLang在M方向的线程块分块大小。
BN = 128  # 设置TileLang在N方向的线程块分块大小。
BK = 32  # 设置TileLang在K方向每次流水迭代处理的分块大小。
CSV_FIELDS = [  # 定义统一CSV的字段顺序。
    "backend",  # 记录执行后端，取值为torch或tilelang。
    "M",  # 记录输入矩阵A的行数。
    "N",  # 记录输入矩阵B的列数。
    "K",  # 记录矩阵乘法的归约维度。
    "dtype",  # 记录输入和输出数据类型。
    "BM",  # 记录TileLang的M方向分块，Torch行留空。
    "BN",  # 记录TileLang的N方向分块，Torch行留空。
    "BK",  # 记录TileLang的K方向分块，Torch行留空。
    "latency_ms",  # 记录单次矩阵乘法的平均GPU延迟，单位为毫秒。
    "tflops",  # 记录按2MNK计算得到的浮点吞吐率，单位为TFLOPS。
]  # 结束CSV字段列表。


@tilelang.jit  # 使用TileLang即时编译装饰器定义FP16 GEMM内核模板。
def tilelang_gemm_fp16(A, B, block_M: int, block_N: int, block_K: int):  # 声明FP16纯矩阵乘法内核。
    M, N_value, K_value = T.const("M, N, K")  # 从编译参数读取固定的矩阵形状。
    dtype = T.float16  # 指定输入矩阵和输出矩阵使用FP16。
    accum_dtype = T.float32  # 指定乘加累积使用FP32，以提高数值稳定性。
    A: T.Tensor((M, K_value), dtype)  # 声明输入矩阵A的形状和数据类型。
    B: T.Tensor((K_value, N_value), dtype)  # 声明输入矩阵B的形状和数据类型。
    C = T.empty((M, N_value), dtype)  # 声明并分配FP16输出矩阵C。
    with T.Kernel(T.ceildiv(N_value, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):  # 为每个输出分块启动一个128线程的GPU线程块。
        A_shared = T.alloc_shared((block_M, block_K), dtype)  # 在共享内存中分配A的当前分块。
        B_shared = T.alloc_shared((block_K, block_N), dtype)  # 在共享内存中分配B的当前分块。
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)  # 在寄存器片段中分配FP32累积结果。
        T.clear(C_local)  # 在开始累积前把局部输出片段清零。
        for ko in T.Pipelined(T.ceildiv(K_value, block_K), num_stages=3):  # 使用三级流水遍历K方向的全部分块。
            T.copy(A[by * block_M, ko * block_K], A_shared)  # 把A的当前全局内存分块复制到共享内存。
            T.copy(B[ko * block_K, bx * block_N], B_shared)  # 把B的当前全局内存分块复制到共享内存。
            T.gemm(A_shared, B_shared, C_local)  # 执行当前分块的Tensor Core矩阵乘加并累积到C_local。
        T.copy(C_local, C[by * block_M, bx * block_N])  # 直接写回GEMM结果，不执行ReLU或其他后处理。
    return C  # 返回完整的FP16输出矩阵。


@tilelang.jit  # 使用TileLang即时编译装饰器定义BF16 GEMM内核模板。
def tilelang_gemm_bf16(A, B, block_M: int, block_N: int, block_K: int):  # 声明BF16纯矩阵乘法内核。
    M, N_value, K_value = T.const("M, N, K")  # 从编译参数读取固定的矩阵形状。
    dtype = T.bfloat16  # 指定输入矩阵和输出矩阵使用BF16。
    accum_dtype = T.float32  # 指定乘加累积使用FP32，以提高数值稳定性。
    A: T.Tensor((M, K_value), dtype)  # 声明输入矩阵A的形状和数据类型。
    B: T.Tensor((K_value, N_value), dtype)  # 声明输入矩阵B的形状和数据类型。
    C = T.empty((M, N_value), dtype)  # 声明并分配BF16输出矩阵C。
    with T.Kernel(T.ceildiv(N_value, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):  # 为每个输出分块启动一个128线程的GPU线程块。
        A_shared = T.alloc_shared((block_M, block_K), dtype)  # 在共享内存中分配A的当前分块。
        B_shared = T.alloc_shared((block_K, block_N), dtype)  # 在共享内存中分配B的当前分块。
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)  # 在寄存器片段中分配FP32累积结果。
        T.clear(C_local)  # 在开始累积前把局部输出片段清零。
        for ko in T.Pipelined(T.ceildiv(K_value, block_K), num_stages=3):  # 使用三级流水遍历K方向的全部分块。
            T.copy(A[by * block_M, ko * block_K], A_shared)  # 把A的当前全局内存分块复制到共享内存。
            T.copy(B[ko * block_K, bx * block_N], B_shared)  # 把B的当前全局内存分块复制到共享内存。
            T.gemm(A_shared, B_shared, C_local)  # 执行当前分块的Tensor Core矩阵乘加并累积到C_local。
        T.copy(C_local, C[by * block_M, bx * block_N])  # 直接写回GEMM结果，不执行ReLU或其他后处理。
    return C  # 返回完整的BF16输出矩阵。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="比较Torch与TileLang纯GEMM的正确性和性能。")  # 创建命令行参数解析器。
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16", help="输入和输出精度。")  # 允许选择FP16或BF16，默认运行FP16。
    parser.add_argument("--warmup", type=int, default=20, help="每个后端正式计时前的预热次数。")  # 设置预热迭代次数。
    parser.add_argument("--iters", type=int, default=100, help="每个后端的正式计时次数。")  # 设置正式计时迭代次数。
    parser.add_argument("--seed", type=int, default=0, help="生成随机输入时使用的种子。")  # 设置随机种子以便复现实验。
    parser.add_argument("--output", type=Path, default=None, help="统一CSV输出路径。")  # 允许覆盖默认结果文件位置。
    args = parser.parse_args()  # 解析终端传入的全部参数。
    if args.warmup < 0:  # 检查预热次数是否合法。
        parser.error("--warmup必须大于或等于0。")  # 对负数预热次数给出明确错误。
    if args.iters <= 0:  # 检查正式计时次数是否合法。
        parser.error("--iters必须大于0。")  # 对非正数计时次数给出明确错误。
    return args  # 返回通过验证的命令行参数。


def get_dtype_config(dtype_name: str):  # 根据字符串名称返回Torch类型、TileLang模板和误差容限。
    if dtype_name == "fp16":  # 处理FP16配置。
        return torch.float16, tilelang_gemm_fp16, 1.0e-2, 1.0e-1  # 为K=4096的不同归约顺序保留相对容限，并允许接近零元素出现有限绝对舍入误差。
    return torch.bfloat16, tilelang_gemm_bf16, 2.0e-2, 5.0e-1  # BF16有效位更少，因此为接近零元素使用更宽但仍有限的绝对容限。


def compile_tilelang_kernel(kernel_definition, m: int):  # 为一个固定shape编译TileLang内核。
    return kernel_definition.compile(  # 调用TileLang编译接口并返回可直接调用的GPU内核。
        M=m,  # 把当前测试的M固化到编译期。
        N=N,  # 把固定的N固化到编译期。
        K=K,  # 把固定的K固化到编译期。
        block_M=BM,  # 把M方向分块大小传给内核模板。
        block_N=BN,  # 把N方向分块大小传给内核模板。
        block_K=BK,  # 把K方向分块大小传给内核模板。
    )  # 完成当前shape的内核编译。


def benchmark_cuda(operation: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:  # 用同一套CUDA Event方法测量两个后端。
    for _ in range(warmup):  # 执行指定次数的预热，排除首次加载和缓存建立影响。
        warmup_output = operation()  # 调用待测操作并保存返回值，确保操作确实被执行。
    torch.cuda.synchronize()  # 等待全部预热操作完成后再开始正式计时。
    start_event = torch.cuda.Event(enable_timing=True)  # 创建能够记录GPU时间戳的起始事件。
    end_event = torch.cuda.Event(enable_timing=True)  # 创建能够记录GPU时间戳的结束事件。
    start_event.record()  # 在当前CUDA流中记录正式计时的起点。
    for _ in range(iters):  # 连续执行正式计时迭代以降低单次测量噪声。
        timed_output = operation()  # 调用待测操作并保存最后一次返回值。
    end_event.record()  # 在当前CUDA流中记录全部正式迭代之后的终点。
    end_event.synchronize()  # 等待结束事件完成，确保所有待测GPU工作已经结束。
    latency_ms = start_event.elapsed_time(end_event) / iters  # 用GPU总耗时除以迭代次数得到平均单次延迟。
    del warmup_output  # 释放预热阶段最后一个输出张量的Python引用。
    del timed_output  # 释放正式计时阶段最后一个输出张量的Python引用。
    return latency_ms  # 返回平均单次GPU延迟，单位为毫秒。


def calculate_tflops(m: int, latency_ms: float) -> float:  # 根据矩阵形状和延迟计算吞吐率。
    operation_count = 2.0 * m * N * K  # GEMM中的一次乘法和一次加法合计按两个浮点操作计算。
    latency_seconds = latency_ms / 1000.0  # 把毫秒转换为秒。
    return operation_count / latency_seconds / 1.0e12  # 把每秒浮点操作数转换为TFLOPS。


def print_performance_row(backend: str, m: int, dtype_name: str, latency_ms: float, tflops: float) -> None:  # 使用“字段: 数据”格式打印一条性能结果。
    print("phase: Performance")  # 表明下面输出属于性能阶段。
    print(f"backend: {backend}")  # 打印当前执行后端。
    print(f"M: {m}")  # 打印当前M维度。
    print(f"N: {N}")  # 打印固定N维度。
    print(f"K: {K}")  # 打印固定K维度。
    print(f"dtype: {dtype_name}")  # 打印当前数据类型。
    if backend == "tilelang":  # 仅为TileLang结果打印分块参数。
        print(f"BM: {BM}")  # 打印M方向分块大小。
        print(f"BN: {BN}")  # 打印N方向分块大小。
        print(f"BK: {BK}")  # 打印K方向分块大小。
    print(f"latency_ms: {latency_ms:.6f}")  # 打印保留六位小数的平均延迟。
    print(f"tflops: {tflops:.6f}")  # 打印保留六位小数的吞吐率。


def make_result_row(backend: str, m: int, dtype_name: str, latency_ms: float, tflops: float) -> dict:  # 构造符合统一CSV模式的一行数据。
    is_tilelang = backend == "tilelang"  # 判断当前行是否来自TileLang后端。
    return {  # 返回字段顺序与CSV_FIELDS一致的结果字典。
        "backend": backend,  # 写入后端名称。
        "M": m,  # 写入当前M维度。
        "N": N,  # 写入固定N维度。
        "K": K,  # 写入固定K维度。
        "dtype": dtype_name,  # 写入数据类型名称。
        "BM": BM if is_tilelang else "",  # TileLang写入BM，Torch保持空字段。
        "BN": BN if is_tilelang else "",  # TileLang写入BN，Torch保持空字段。
        "BK": BK if is_tilelang else "",  # TileLang写入BK，Torch保持空字段。
        "latency_ms": f"{latency_ms:.6f}",  # 以固定六位小数写入平均延迟。
        "tflops": f"{tflops:.6f}",  # 以固定六位小数写入TFLOPS。
    }  # 完成一行统一结果。


def write_results(output_path: Path, rows: list[dict]) -> None:  # 把全部已通过正确性门禁的性能结果一次性写入CSV。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保kernel/results目录存在。
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:  # 以覆盖模式创建UTF-8 CSV，避免重复旧结果。
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)  # 创建使用统一字段顺序的字典写入器。
        writer.writeheader()  # 写入CSV表头。
        writer.writerows(rows)  # 写入本次运行得到的全部Torch和TileLang结果。


def main() -> None:  # 定义程序主入口。
    args = parse_args()  # 读取并验证命令行参数。
    if not torch.cuda.is_available():  # 检查PyTorch能否访问CUDA设备。
        raise RuntimeError("没有检测到可用CUDA设备，无法运行GPU GEMM基准测试。")  # 没有GPU时立即停止并说明原因。
    torch.manual_seed(args.seed)  # 设置CPU随机种子，保持实验配置完整。
    torch.cuda.manual_seed_all(args.seed)  # 设置全部CUDA设备的随机种子，确保输入可以复现。
    torch_dtype, kernel_definition, rtol, atol = get_dtype_config(args.dtype)  # 取得当前精度对应的完整配置。
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():  # 检查当前GPU是否原生支持BF16。
        raise RuntimeError("当前CUDA设备不支持BF16，请改用--dtype fp16。")  # 在不支持BF16时给出明确提示。
    default_output = Path(__file__).resolve().parents[1] / "results" / "gemm_benchmark_results.csv"  # 构造不含用户名或主机地址的默认项目内结果路径。
    output_path = args.output if args.output is not None else default_output  # 优先使用用户显式指定的CSV路径。
    rows = []  # 创建列表，用于暂存通过正确性验证后的全部性能结果。
    print(f"device: {torch.cuda.get_device_name(0)}")  # 打印实际参与测试的GPU名称。
    print(f"dtype: {args.dtype}")  # 打印本次基准测试的数据类型。
    print(f"warmup: {args.warmup}")  # 打印每个后端的预热次数。
    print(f"iters: {args.iters}")  # 打印每个后端的正式计时次数。
    for m in M_LIST:  # 按顺序测试全部指定M维度。
        a = torch.randn((m, K), device="cuda", dtype=torch_dtype)  # 在GPU上生成当前shape的输入矩阵A。
        b = torch.randn((K, N), device="cuda", dtype=torch_dtype)  # 在GPU上生成所有后端共用的输入矩阵B。
        tilelang_kernel = compile_tilelang_kernel(kernel_definition, m)  # 在计时前编译当前shape的TileLang内核。
        print("phase: Correctness")  # 明确表明当前先执行正确性验证。
        print("backend: tilelang")  # 表明被验证的实现是TileLang内核。
        print(f"M: {m}")  # 打印当前验证的M维度。
        print(f"N: {N}")  # 打印当前验证的N维度。
        print(f"K: {K}")  # 打印当前验证的K维度。
        print(f"dtype: {args.dtype}")  # 打印当前验证的数据类型。
        ref = a @ b  # 使用PyTorch矩阵乘法计算同一输入的参考结果。
        out = tilelang_kernel(a, b)  # 使用TileLang纯GEMM内核计算同一输入的待验证结果。
        torch.cuda.synchronize()  # 等待两次GPU计算完成后再比较输出。
        try:  # 捕获数值比较失败，以便留下明确的失败状态。
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)  # 使用适合FP16或BF16的容限比较，不要求逐位相同。
        except AssertionError:  # 处理当前shape的正确性验证失败。
            print("correctness = FAIL")  # 打印机器和人都容易识别的失败标记。
            raise  # 立即终止程序，保证错误结果不会进入性能测试和CSV。
        print("correctness = PASS")  # 只有数值比较通过后才打印要求保留的成功标记。
        del ref  # 释放参考输出，避免它影响后续性能计时的显存占用。
        del out  # 释放TileLang验证输出，避免它影响后续性能计时的显存占用。

        def torch_operation() -> torch.Tensor:  # 定义使用当前同一组输入的Torch待测操作。
            return torch.matmul(a, b)  # 执行不带任何后处理的Torch纯矩阵乘法。

        def tilelang_operation() -> torch.Tensor:  # 定义使用当前同一组输入的TileLang待测操作。
            return tilelang_kernel(a, b)  # 执行不带ReLU的TileLang纯矩阵乘法。

        torch_latency_ms = benchmark_cuda(torch_operation, args.warmup, args.iters)  # 在正确性通过后用统一方法测量Torch延迟。
        torch_tflops = calculate_tflops(m, torch_latency_ms)  # 计算Torch在当前shape上的TFLOPS。
        print_performance_row("torch", m, args.dtype, torch_latency_ms, torch_tflops)  # 按字段格式打印Torch性能。
        rows.append(make_result_row("torch", m, args.dtype, torch_latency_ms, torch_tflops))  # 暂存Torch统一CSV行。
        tilelang_latency_ms = benchmark_cuda(tilelang_operation, args.warmup, args.iters)  # 用完全相同的方法测量TileLang延迟。
        tilelang_tflops = calculate_tflops(m, tilelang_latency_ms)  # 计算TileLang在当前shape上的TFLOPS。
        print_performance_row("tilelang", m, args.dtype, tilelang_latency_ms, tilelang_tflops)  # 按字段格式打印TileLang性能。
        rows.append(make_result_row("tilelang", m, args.dtype, tilelang_latency_ms, tilelang_tflops))  # 暂存TileLang统一CSV行。
        del a  # 释放当前shape的输入矩阵A。
        del b  # 释放当前shape的输入矩阵B。
        del tilelang_kernel  # 释放当前shape的已编译内核Python引用。
    write_results(output_path, rows)  # 全部shape完成后一次性写出统一CSV。
    print(f"csv: {output_path}")  # 打印最终CSV文件位置。
    print(f"rows: {len(rows)}")  # 打印CSV中的性能数据行数。


if __name__ == "__main__":  # 仅在直接运行本文件时启动基准测试。
    main()  # 调用主函数执行正确性验证、性能测试和CSV写入。
