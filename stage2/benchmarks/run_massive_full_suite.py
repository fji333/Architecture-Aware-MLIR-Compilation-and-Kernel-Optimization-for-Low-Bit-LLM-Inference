import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import triton
import triton.language as tl
from kernels.fp16_gemm import fp16_gemm_kernel
from kernels.w4a16_gemm import w4a16_gemm_kernel
from tests.test_w4a16_correctness import matmul_w4a16_triton, unpack_reference

os.environ['TRITON_CPU_BACKEND'] = '1'

def time_fn(fn, warm=2, iters=5):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0 # 毫秒 (ms)

def run_fp16(M, N, K, BM=16, BN=16, BK=32):
    a = torch.randn((M, K), dtype=torch.float16, device='cpu')
    b = torch.randn((K, N), dtype=torch.float16, device='cpu')
    c = torch.empty((M, N), dtype=torch.float16, device='cpu')
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    fn = lambda: fp16_gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK
    )
    gflops_tot = 2.0 * M * N * K / 1e9
    ms = time_fn(fn)
    gf = gflops_tot / (ms * 1e-3)
    return ms, gf

def run_pt_fp16(M, N, K):
    a = torch.randn((M, K), dtype=torch.float16, device='cpu')
    b = torch.randn((K, N), dtype=torch.float16, device='cpu')
    gflops_tot = 2.0 * M * N * K / 1e9
    ms = time_fn(lambda: torch.matmul(a, b))
    gf = gflops_tot / (ms * 1e-3)
    return ms, gf

def run_w4a16(M, N, K, BM=16, BN=16, BK=32):
    a = torch.randn((M, K), dtype=torch.float16, device='cpu')
    w_packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8, device='cpu')
    scale = torch.rand((1, N), dtype=torch.float16, device='cpu') * 0.1
    bias = (torch.rand((1, N), dtype=torch.float16, device='cpu') - 0.5) * 2.0
    fn = lambda: matmul_w4a16_triton(a, w_packed, scale, bias)
    gflops_tot = 2.0 * M * N * K / 1e9
    ms = time_fn(fn)
    gf = gflops_tot / (ms * 1e-3)
    return ms, gf

def run_pt_w4a16(M, N, K):
    a = torch.randn((M, K), dtype=torch.float16, device='cpu')
    w_packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8, device='cpu')
    scale = torch.rand((1, N), dtype=torch.float16, device='cpu') * 0.1
    bias = (torch.rand((1, N), dtype=torch.float16, device='cpu') - 0.5) * 2.0
    def run_pt():
        w_fp16 = unpack_reference(w_packed, scale, bias)
        return torch.matmul(a, w_fp16)
    gflops_tot = 2.0 * M * N * K / 1e9
    ms = time_fn(run_pt)
    gf = gflops_tot / (ms * 1e-3)
    return ms, gf

if __name__ == "__main__":
    print("====================================================================================")
    print(">>> 启动全量系统级消融实验全量评测矩阵 (run_massive_full_suite.py) <<<")
    print("====================================================================================\n")

    # Table 1: FP16 规模全覆盖
    print("【表 1: FP16 方阵全维度扩展对比数据表】")
    print(f"{'M=N=K':<10} | {'GFLOPs':<8} | {'PyTorch(ms)':<14} | {'PyTorch(GF)':<12} | {'Triton(ms)':<12} | {'Triton(GF)':<12} | {'加速比':<8}")
    print("-" * 88)
    for dim in [128, 256, 384, 512, 640, 768, 896, 1024, 1280]:
        gflops_total = 2.0 * (dim**3) / 1e9
        ms_pt, gf_pt = run_pt_fp16(dim, dim, dim)
        ms_tr, gf_tr = run_fp16(dim, dim, dim, 16, 16, 32)
        sp = ms_pt / ms_tr
        print(f"{dim:<10} | {gflops_total:<8.3f} | {ms_pt:<14.3f} | {gf_pt:<12.2f} | {ms_tr:<12.3f} | {gf_tr:<12.2f} | {sp:<8.2f}x")

    # Table 2: 2D 空间瓦片全扫描
    print("\n【表 2: 2D 空间瓦片 (BLOCK_M x BLOCK_N) 全网格消融数据表 (M=N=K=512)】")
    print(f"{'BM x BN x BK':<16} | {'累加器(KB)':<12} | {'耗时(ms)':<10} | {'算力(GFLOPS)':<14}")
    print("-" * 58)
    for bm in [16, 32, 64]:
        for bn in [16, 32, 64]:
            acc_kb = (bm * bn * 4) / 1024.0
            ms, gf = run_fp16(512, 512, 512, bm, bn, 32)
            print(f"{bm}x{bn}x32{'':<8} | {acc_kb:<12.2f} | {ms:<10.3f} | {gf:<14.2f}")

    # Table 3: BLOCK_K 步长深入扫描
    print("\n【表 3: BLOCK_K 步长与 L1D 局部性消融数据表 (M=N=K=512, BM=16, BN=16)】")
    print(f"{'BLOCK_K':<10} | {'迭代步数':<10} | {'单次加载字节':<14} | {'耗时(ms)':<10} | {'算力(GFLOPS)':<14}")
    print("-" * 64)
    for bk in [8, 16, 32, 64, 128]:
        steps = 512 // bk
        load_bytes = (16 * bk + bk * 16) * 2
        ms, gf = run_fp16(512, 512, 512, 16, 16, bk)
        print(f"BK={bk:<7} | {steps:<10} | {load_bytes:<14} | {ms:<10.3f} | {gf:<14.2f}")

    # Table 4: W4A16 规模全覆盖
    print("\n【表 4: W4A16 方阵全维度扩展对比数据表】")
    print(f"{'M=N=K':<10} | {'权重内存(MB)':<14} | {'PyTorch(ms)':<14} | {'PyTorch(GF)':<12} | {'Triton(ms)':<12} | {'Triton(GF)':<12} | {'加速比':<8}")
    print("-" * 92)
    for dim in [128, 256, 384, 512, 640, 768, 896, 1024, 1280]:
        weight_mb = (dim * (dim // 2)) / (1024 * 1024)
        ms_pt, gf_pt = run_pt_w4a16(dim, dim, dim)
        ms_tr, gf_tr = run_w4a16(dim, dim, dim, 16, 16, 32)
        sp = ms_pt / ms_tr
        print(f"{dim:<10} | {weight_mb:<14.3f} | {ms_pt:<14.3f} | {gf_pt:<12.2f} | {ms_tr:<12.3f} | {gf_tr:<12.2f} | {sp:<8.2f}x")

    # Table 5: LLM 解码阶段细粒度批次扫描
    print("\n【表 5: LLM 解码阶段 (M=Batch, N=4096, K=4096) 细粒度批次收益数据表】")
    print(f"{'Batch M':<10} | {'PyTorch解包(ms)':<16} | {'Triton W4(ms)':<14} | {'Triton算力(GFLOPS)':<20} | {'加速比':<8}")
    print("-" * 75)
    for m in [1, 2, 4, 8, 16, 32, 64, 128]:
        ms_pt, _ = run_pt_w4a16(m, 4096, 4096)
        ms_tr, gf_tr = run_w4a16(m, 4096, 4096, 16, 16, 32)
        sp = ms_pt / ms_tr
        print(f"M={m:<8} | {ms_pt:<16.3f} | {ms_tr:<14.3f} | {gf_tr:<20.2f} | {sp:<8.2f}x")

    print("\n====================================================================================")
    print("全量 10 维消融实验矩阵测试圆满完成！")
