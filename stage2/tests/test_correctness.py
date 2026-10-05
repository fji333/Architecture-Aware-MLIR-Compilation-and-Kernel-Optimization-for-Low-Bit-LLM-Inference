import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
from kernels.fp16_gemm import fp16_gemm_kernel

def matmul_triton(a, b):
    # 强制验证内存连续性（你项目一里最痛的领悟）
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions"
    
    # 分配输出矩阵的内存 (FP16)
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # 极其致命的超参数：Block Tiling 尺寸
    # 如果这些值设得太大，就会打断 Apple M4 的 DMP 预取器，引发 L1 击穿
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    # 一维 Grid 启动网格
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    # 核函数点火发射（向编译器下达编译和执行指令）
    fp16_gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c

if __name__ == "__main__":
    print("🚀 准备启动 Triton CPU 测试...")
    torch.manual_seed(0)
    
    M, N, K = 512, 512, 512
    # 注意：设备目前写的是 'cpu'，这在传统 Triton 里是会直接报错的！
    a = torch.randn((M, K), dtype=torch.float16, device='cpu')
    b = torch.randn((K, N), dtype=torch.float16, device='cpu')
    
    try:
        triton_output = matmul_triton(a, b)
        torch_output = torch.matmul(a, b)
        
        # 精度验证
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
            print("✅ 验证通过！Triton 生成的机器码计算结果与 PyTorch 矩阵乘法完全一致！")
        else:
            print("❌ 验证失败！结果有误差，你的 Stride 或 Mask 逻辑可能有漏洞！")
            
    except Exception as e:
        print(f"❌ 运行崩溃: {e}")
        print("💡 导师提示：如果报错说不支持 CPU 或者找不到后端，说明你的编译器环境还没搭！")
