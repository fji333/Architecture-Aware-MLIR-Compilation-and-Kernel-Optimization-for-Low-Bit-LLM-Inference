import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
import triton.language as tl
from kernels.fp16_gemm import fp16_gemm_kernel

def matmul_triton(a, b):
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    fp16_gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'], 
        x_vals=[128 * i for i in range(2, 11)], 
        line_arg='provider',      
        line_vals=['pytorch', 'triton'], 
        line_names=['PyTorch', 'Triton (M4 CPU)'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel='TFLOPS',          
        plot_name='fp16-gemm-performance',
        args={},  
    )
)
def benchmark(M, N, K, provider):
    # TODO 1: 初始化输入矩阵 A 和 B (注意必须指定 device='cpu')
    a = torch.randn((M, K), device='cpu', dtype=torch.float16)
    b = torch.randn((K, N), device='cpu', dtype=torch.float16)
    
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'pytorch':
        # TODO 2: 使用 lambda 匿名函数封装 PyTorch 的底层 C++ 算子
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)

    if provider == 'triton':
        # TODO 3: 使用 lambda 匿名函数封装我们自己编译的 Triton 算子
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_triton(a, b), quantiles=quantiles)

    # TODO 4: 性能转化数学公式推导
    # 总浮点计算量 (FLOPs) = 2 * M * N * K
    # 将时间 (ms) 转换为秒 (s): ms * 1e-3
    # 算力 (TFLOPS) = FLOPs / 秒 / 1e12 = (2 * M * N * K) / (ms * 1e-3) / 1e12 = (2 * M * N * K) / (ms * 1e9)
    perf = lambda t: 2 * M * N * K / (t * 1e9)
    
    # Triton perf_report 要求返回 (中位数TFLOPS, 最高TFLOPS, 最低TFLOPS)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    # 屏蔽 Triton 寻找 GPU 的警告
    import os
    os.environ['TRITON_CPU_BACKEND'] = '1'
    benchmark.run(show_plots=True, print_data=True)
