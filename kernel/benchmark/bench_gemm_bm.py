"""固定GEMM形状并扫描TileLang的BM分块参数。"""  # 说明本脚本只研究BM变化对FP16纯GEMM性能的影响。

import argparse  # 导入命令行参数解析模块，用于配置预热和正式计时次数。
import csv  # 导入CSV模块，用于保存每组BM配置的正确性和性能结果。
from pathlib import Path  # 导入路径对象，用于构造项目内的结果文件路径。
from typing import Callable  # 导入可调用对象类型，用于标注统一GPU计时函数。

import torch  # 导入PyTorch，用于生成输入、计算参考结果和创建CUDA Event。
import tilelang  # 导入TileLang，用于定义和编译FP16 GEMM内核。
import tilelang.language as T  # 导入TileLang语言接口，并使用T作为简短别名。


M = 1024  # 固定输入矩阵A的行数为1024。
N = 4096  # 固定输入矩阵B的列数为4096。
K = 4096  # 固定矩阵乘法的归约维度为4096。
BM_LIST = [32, 64, 128]  # 只改变M方向分块大小，并按从小到大的顺序测试。
BN = 128  # 固定N方向分块大小为128。
BK = 32  # 固定K方向分块大小为32。
NUM_THREADS = 128  # 固定每个CUDA线程块使用128个线程。
NUM_STAGES = 3  # 固定K方向异步流水线使用3个阶段。
DTYPE_NAME = "fp16"  # 固定输入和输出数据类型名称为FP16。
TORCH_DTYPE = torch.float16  # 固定PyTorch输入和参考结果使用FP16。
RTOL = 1.0e-2  # 设置FP16结果比较使用的相对误差容限。
ATOL = 1.0e-1  # 设置K等于4096时允许的有限绝对舍入误差。
CSV_FIELDS = [  # 定义用户要求的CSV字段和固定顺序。
    "BM",  # 记录M方向分块大小。
    "BN",  # 记录N方向分块大小。
    "BK",  # 记录K方向分块大小。
    "num_threads",  # 记录每个CUDA线程块的线程数量。
    "num_stages",  # 记录TileLang流水线阶段数量。
    "latency_ms",  # 记录单次TileLang GEMM平均GPU延迟，单位为毫秒。
    "TFLOPS",  # 记录根据2MNK计算的浮点吞吐率。
    "correctness",  # 记录当前配置的正确性验证结果。
]  # 结束CSV字段列表。


@tilelang.jit  # 使用TileLang即时编译装饰器定义可配置分块的FP16 GEMM模板。
def tilelang_gemm(A, B, block_M: int, block_N: int, block_K: int, num_threads: int, num_stages: int):  # 声明纯GEMM内核及其编译期参数。
    matrix_m, matrix_n, matrix_k = T.const("M, N, K")  # 从编译参数读取固定矩阵形状。
    dtype = T.float16  # 指定输入矩阵和输出矩阵使用FP16。
    accum_dtype = T.float32  # 指定Tensor Core乘加结果使用FP32累积。
    A: T.Tensor((matrix_m, matrix_k), dtype)  # 声明输入矩阵A的形状和数据类型。
    B: T.Tensor((matrix_k, matrix_n), dtype)  # 声明输入矩阵B的形状和数据类型。
    C = T.empty((matrix_m, matrix_n), dtype)  # 分配保存纯GEMM结果的FP16输出矩阵。
    with T.Kernel(T.ceildiv(matrix_n, block_N), T.ceildiv(matrix_m, block_M), threads=num_threads) as (bx, by):  # 为每个输出分块启动一个CUDA线程块。
        A_shared = T.alloc_shared((block_M, block_K), dtype)  # 在共享内存中分配A的当前分块。
        B_shared = T.alloc_shared((block_K, block_N), dtype)  # 在共享内存中分配B的当前分块。
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)  # 在寄存器片段中分配FP32累积结果。
        T.clear(C_local)  # 在第一次乘加前把局部累积结果清零。
        for ko in T.Pipelined(T.ceildiv(matrix_k, block_K), num_stages=num_stages):  # 使用固定阶段数的流水线遍历全部K分块。
            T.copy(A[by * block_M, ko * block_K], A_shared)  # 把A的当前全局内存分块复制到共享内存。
            T.copy(B[ko * block_K, bx * block_N], B_shared)  # 把B的当前全局内存分块复制到共享内存。
            T.gemm(A_shared, B_shared, C_local)  # 执行当前分块的Tensor Core矩阵乘加。
        T.copy(C_local, C[by * block_M, bx * block_N])  # 直接写回纯GEMM结果，不执行ReLU或其他后处理。
    return C  # 返回完整的FP16输出矩阵。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="固定GEMM形状并扫描TileLang的BM参数。")  # 创建命令行参数解析器。
    parser.add_argument("--warmup", type=int, default=20, help="每组BM配置正式计时前的预热次数。")  # 设置默认预热次数为20。
    parser.add_argument("--iters", type=int, default=100, help="每组BM配置的正式计时次数。")  # 设置默认正式计时次数为100。
    parser.add_argument("--seed", type=int, default=0, help="生成所有配置共用输入时的随机种子。")  # 设置随机种子以便复现实验。
    parser.add_argument("--output", type=Path, default=None, help="BM扫描结果CSV的输出路径。")  # 允许覆盖默认CSV路径。
    args = parser.parse_args()  # 解析终端传入的命令行参数。
    if args.warmup < 0:  # 检查预热次数是否有效。
        parser.error("--warmup必须大于或等于0。")  # 对负数预热次数给出明确错误。
    if args.iters <= 0:  # 检查正式计时次数是否有效。
        parser.error("--iters必须大于0。")  # 对非正数计时次数给出明确错误。
    return args  # 返回已经验证的命令行参数。


def compile_kernel(block_m: int):  # 为当前BM配置编译TileLang内核。
    return tilelang_gemm.compile(  # 调用TileLang编译接口并返回可直接执行的GPU内核。
        M=M,  # 把固定M维度传入编译器。
        N=N,  # 把固定N维度传入编译器。
        K=K,  # 把固定K维度传入编译器。
        block_M=block_m,  # 把当前扫描的BM传入编译器。
        block_N=BN,  # 把固定BN传入编译器。
        block_K=BK,  # 把固定BK传入编译器。
        num_threads=NUM_THREADS,  # 把固定线程数传入编译器。
        num_stages=NUM_STAGES,  # 把固定流水线阶段数传入编译器。
    )  # 完成当前BM配置的内核编译。


def benchmark_cuda(operation: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:  # 使用CUDA Event测量平均GPU延迟。
    for _ in range(warmup):  # 执行预热迭代以排除首次运行和缓存建立影响。
        warmup_output = operation()  # 执行一次待测TileLang内核并保留最后一个预热输出引用。
    torch.cuda.synchronize()  # 等待全部预热任务完成后再开始正式计时。
    start_event = torch.cuda.Event(enable_timing=True)  # 创建能够记录GPU时间戳的起始事件。
    end_event = torch.cuda.Event(enable_timing=True)  # 创建能够记录GPU时间戳的结束事件。
    start_event.record()  # 在当前CUDA流中记录正式计时起点。
    for _ in range(iters):  # 连续执行指定次数以降低单次测量噪声。
        timed_output = operation()  # 执行一次待测TileLang内核并保留最后一个计时输出引用。
    end_event.record()  # 在当前CUDA流中记录全部正式迭代后的终点。
    end_event.synchronize()  # 等待结束事件完成，确保GPU工作已经执行完毕。
    latency_ms = start_event.elapsed_time(end_event) / iters  # 用GPU总耗时除以迭代次数得到平均延迟。
    del warmup_output  # 释放最后一个预热输出的Python引用。
    del timed_output  # 释放最后一个正式计时输出的Python引用。
    return latency_ms  # 返回单位为毫秒的平均GPU延迟。


def calculate_tflops(latency_ms: float) -> float:  # 根据固定shape和平均延迟计算TFLOPS。
    operation_count = 2.0 * M * N * K  # 按一次乘法和一次加法计算GEMM总浮点操作数。
    latency_seconds = latency_ms / 1000.0  # 把毫秒转换为秒。
    return operation_count / latency_seconds / 1.0e12  # 把每秒浮点操作数转换为TFLOPS。


def make_result_row(block_m: int, latency_ms: float, tflops: float) -> dict:  # 构造一条符合指定字段的CSV结果。
    return {  # 返回当前BM配置的完整记录。
        "BM": block_m,  # 写入当前M方向分块大小。
        "BN": BN,  # 写入固定N方向分块大小。
        "BK": BK,  # 写入固定K方向分块大小。
        "num_threads": NUM_THREADS,  # 写入固定CUDA线程数。
        "num_stages": NUM_STAGES,  # 写入固定流水线阶段数。
        "latency_ms": f"{latency_ms:.6f}",  # 以六位小数写入平均延迟。
        "TFLOPS": f"{tflops:.6f}",  # 以六位小数写入吞吐率。
        "correctness": "PASS",  # 只有通过assert_close后才会创建结果行。
    }  # 完成当前配置的结果记录。


def write_results(output_path: Path, rows: list[dict]) -> None:  # 把全部通过正确性验证的配置一次性写入CSV。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保项目内的results目录存在。
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:  # 以覆盖模式创建UTF-8 CSV文件。
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)  # 创建使用指定字段顺序的CSV写入器。
        writer.writeheader()  # 写入用户要求的八个字段名称。
        writer.writerows(rows)  # 写入BM等于32、64和128的全部结果。


def main() -> None:  # 定义程序主入口。
    args = parse_args()  # 读取并验证命令行参数。
    if not torch.cuda.is_available():  # 检查当前PyTorch环境能否访问CUDA设备。
        raise RuntimeError("没有检测到可用CUDA设备，无法运行BM扫描实验。")  # 没有GPU时立即停止并说明原因。
    torch.manual_seed(args.seed)  # 设置CPU随机种子，保持实验配置完整。
    torch.cuda.manual_seed_all(args.seed)  # 设置CUDA随机种子，确保测试输入可以复现。
    default_output = Path(__file__).resolve().parents[1] / "results" / "tilelang_gemm_bm_results.csv"  # 构造项目内默认CSV路径。
    output_path = args.output if args.output is not None else default_output  # 优先使用用户显式指定的输出路径。
    a = torch.randn((M, K), device="cuda", dtype=TORCH_DTYPE)  # 创建所有BM配置共用的FP16输入矩阵A。
    b = torch.randn((K, N), device="cuda", dtype=TORCH_DTYPE)  # 创建所有BM配置共用的FP16输入矩阵B。
    ref = a @ b  # 使用torch.matmul计算一次所有配置共用的FP16参考结果。
    rows = []  # 创建列表，用于暂存全部通过正确性验证的性能结果。
    print(f"device: {torch.cuda.get_device_name(0)}")  # 打印实际执行实验的GPU名称。
    print(f"M: {M}")  # 打印固定M维度。
    print(f"N: {N}")  # 打印固定N维度。
    print(f"K: {K}")  # 打印固定K维度。
    print(f"dtype: {DTYPE_NAME}")  # 打印固定数据类型。
    print(f"warmup: {args.warmup}")  # 打印每组配置的预热次数。
    print(f"iters: {args.iters}")  # 打印每组配置的正式计时次数。
    for block_m in BM_LIST:  # 依次测试BM等于32、64和128的配置。
        tilelang_kernel = compile_kernel(block_m)  # 在计时外编译当前BM配置的TileLang内核。
        print("phase: Correctness")  # 表明当前首先执行正确性验证。
        print(f"BM: {block_m}")  # 打印当前扫描的BM。
        print(f"BN: {BN}")  # 打印固定BN。
        print(f"BK: {BK}")  # 打印固定BK。
        print(f"num_threads: {NUM_THREADS}")  # 打印固定CUDA线程数。
        print(f"num_stages: {NUM_STAGES}")  # 打印固定流水线阶段数。
        out = tilelang_kernel(a, b)  # 使用当前BM配置计算与参考实现相同输入的结果。
        torch.cuda.synchronize()  # 等待参考结果和TileLang结果全部完成后再比较。
        try:  # 捕获数值比较失败，以便留下明确状态。
            torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)  # 使用FP16容差比较，不要求逐位相同。
        except AssertionError:  # 处理当前BM配置的正确性失败。
            print("correctness = FAIL")  # 打印明确的正确性失败标记。
            raise  # 立即终止，禁止错误配置进入性能计时和CSV。
        print("correctness = PASS")  # 只有当前配置验证通过后才打印成功标记。
        del out  # 释放正确性验证输出，避免影响后续性能计时显存占用。

        def tilelang_operation() -> torch.Tensor:  # 定义使用当前配置和共用输入的待测操作。
            return tilelang_kernel(a, b)  # 执行不带ReLU的TileLang纯GEMM。

        latency_ms = benchmark_cuda(tilelang_operation, args.warmup, args.iters)  # 正确性通过后才测量当前配置的平均延迟。
        tflops = calculate_tflops(latency_ms)  # 根据固定shape和实测延迟计算TFLOPS。
        print("phase: Performance")  # 表明下面输出属于性能阶段。
        print(f"BM: {block_m}")  # 打印当前M方向分块大小。
        print(f"BN: {BN}")  # 打印固定N方向分块大小。
        print(f"BK: {BK}")  # 打印固定K方向分块大小。
        print(f"num_threads: {NUM_THREADS}")  # 打印固定CUDA线程数。
        print(f"num_stages: {NUM_STAGES}")  # 打印固定流水线阶段数。
        print(f"latency_ms: {latency_ms:.6f}")  # 打印保留六位小数的平均延迟。
        print(f"TFLOPS: {tflops:.6f}")  # 打印保留六位小数的吞吐率。
        print("correctness: PASS")  # 在性能记录旁再次保留当前配置的正确性状态。
        rows.append(make_result_row(block_m, latency_ms, tflops))  # 把当前配置的完整结果暂存到列表。
        del tilelang_kernel  # 释放当前配置的已编译内核Python引用。
    write_results(output_path, rows)  # 全部配置成功后一次性覆盖写入统一CSV。
    print(f"csv: {output_path}")  # 打印最终CSV文件位置。
    print(f"rows: {len(rows)}")  # 打印CSV中的配置行数，应为3。


if __name__ == "__main__":  # 仅在直接运行本文件时启动BM扫描实验。
    main()  # 执行正确性验证、性能测试和CSV写入。
