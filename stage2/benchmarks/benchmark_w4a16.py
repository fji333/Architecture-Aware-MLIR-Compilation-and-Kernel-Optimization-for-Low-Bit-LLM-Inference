import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
import triton.language as tl
from kernels.w4a16_gemm import w4a16_gemm_kernel
from tests.test_w4a16_correctness import matmul_w4a16_triton, unpack_reference

os.environ['TRITON_CPU_BACKEND'] = '1'

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[256 * i for i in range(1, 6)],
        line_arg='provider',
        line_vals=['pytorch_dequant', 'triton_w4a16'],
        line_names=['PyTorch (Dequant + GEMM)', 'Triton W4A16 (M4 JIT)'],
        styles=[('red', '-'), ('blue', '-')],
        ylabel='TFLOPS',
        plot_name='w4a16-gemm-performance',
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), dtype=torch.float16, device='cpu')
    w_packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8, device='cpu')
    scale = torch.rand((1, N), dtype=torch.float16, device='cpu') * 0.1
    bias = (torch.rand((1, N), dtype=torch.float16, device='cpu') - 0.5) * 2.0

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'pytorch_dequant':
        # 模拟工业界标准做法: 运行时将 4-bit 权重解包后再调用 PyTorch matmul
        def run_pytorch():
            w_fp16 = unpack_reference(w_packed, scale, bias)
            return torch.matmul(a, w_fp16)
        ms, min_ms, max_ms = triton.testing.do_bench(run_pytorch, quantiles=quantiles)

    if provider == 'triton_w4a16':
        # Triton 融合算子: 内存仅传输 4-bit 权重，在 SIMD 寄存器内部就地解包并累加
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_w4a16_triton(a, w_packed, scale, bias), 
            quantiles=quantiles
        )

    # 矩阵等效 FLOPs = 2 * M * N * K
    perf = lambda t: (2 * M * N * K) / (t * 1e9)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    print("================================================================================")
    print(">>> 启动 W4A16 算子端到端性能压测 (PyTorch vs Triton M4 JIT) <<<")
    print("================================================================================")
    benchmark.run(show_plots=True, print_data=True)
