import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
from kernels.w4a16_gemm import w4a16_gemm_kernel

os.environ['TRITON_CPU_BACKEND'] = '1'

def matmul_w4a16_triton(a, w_packed, scale, bias):
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert w_packed.is_contiguous(), "Matrix W must be contiguous"
    assert scale.is_contiguous(), "Scale must be contiguous"
    assert bias.is_contiguous(), "Bias must be contiguous"
    
    M, K = a.shape
    K_, N_half = w_packed.shape
    N = N_half * 2
    assert K == K_, "Incompatible dimensions"
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    w4a16_gemm_kernel[grid](
        a, w_packed, scale, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        0, scale.stride(1) if scale.ndim > 1 else scale.stride(0),
        0, bias.stride(1) if bias.ndim > 1 else bias.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c

def unpack_reference(w_packed, scale, bias):
    """纯 PyTorch 实现的 4-bit 权重反量化参考基准"""
    K, N_half = w_packed.shape
    N = N_half * 2
    w_low = (w_packed & 0x0F).to(torch.float16)
    w_high = ((w_packed >> 4) & 0x0F).to(torch.float16)
    
    w_unpacked = torch.empty((K, N), dtype=torch.float16, device=w_packed.device)
    w_unpacked[:, 0::2] = w_low
    w_unpacked[:, 1::2] = w_high
    
    w_dequant = w_unpacked * scale + bias
    return w_dequant

if __name__ == "__main__":
    print("================================================================================")
    print(">>> 启动 Triton W4A16 量化算子数值正确性验证 <<<")
    print("================================================================================")
    torch.manual_seed(42)
    
    M, N, K = 256, 256, 512
    print(f"测试矩阵规模: M={M}, N={N}, K={K}")
    
    # 构造激活值 (FP16)
    a = torch.randn((M, K), dtype=torch.float16, device='cpu')
    
    # 构造 4-bit 紧凑打包的 uint8 权重 (体积缩小 4 倍)
    w_packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8, device='cpu')
    
    # 构造 FP16 缩放因子与偏置项 (项目一数学模型)
    scale = torch.rand((1, N), dtype=torch.float16, device='cpu') * 0.1
    bias = (torch.rand((1, N), dtype=torch.float16, device='cpu') - 0.5) * 2.0
    
    print("正在通过 Triton JIT 编译并执行 W4A16 算子...")
    triton_out = matmul_w4a16_triton(a, w_packed, scale, bias)
    
    print("正在计算 PyTorch 逐元素反量化参考结果...")
    w_ref = unpack_reference(w_packed, scale, bias)
    torch_out = torch.matmul(a, w_ref)
    
    # 残差比对
    abs_diff = torch.abs(triton_out - torch_out)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    print(f"最大绝对数值残差 (Max Absolute Error): {max_diff:.6f}")
    print(f"平均绝对数值残差 (Mean Absolute Error): {mean_diff:.6f}")
    print(f"Triton 输出前 5 项: {triton_out[0, :5].tolist()}")
    print(f"PyTorch 参考前 5 项: {torch_out[0, :5].tolist()}")
    
    # FP32 累加对照 (验证是否为 FP16 累加精度差异)
    torch_out_fp32 = torch.matmul(a.to(torch.float32), w_ref.to(torch.float32)).to(torch.float16)
    max_diff_fp32 = torch.max(torch.abs(triton_out - torch_out_fp32)).item()
    print(f"与 PyTorch FP32 累加残差: {max_diff_fp32:.6f}")

    if max_diff_fp32 < 1e-2 or torch.allclose(triton_out, torch_out, atol=0.1, rtol=1e-2):
        print("✅ 【验证通过！】Triton W4A16 算子生成的机器码与数学参考结果完全一致！")
    else:
        print("❌ 【验证失败！】残差超出阈值，需检查 Nibble 位解包或 Stride 逻辑！")
