#!/usr/bin/env python3
# ==============================================================================
# Project 3: Rigorous Correctness Test for MLIR W4A16 GEMM on Apple M4
# Sweeps multiple matrix shapes and tests residual accuracy against PyTorch
# ==============================================================================

import ctypes
import os
import sys
import torch
import numpy as np

# MemRef Descriptor Structures according to MLIR LLVM C-ABI
class StridedMemRef2D(ctypes.Structure):
    _fields_ = [
        ("allocated_ptr", ctypes.c_void_p),
        ("aligned_ptr", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]

class StridedMemRef1D(ctypes.Structure):
    _fields_ = [
        ("allocated_ptr", ctypes.c_void_p),
        ("aligned_ptr", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 1),
        ("strides", ctypes.c_int64 * 1),
    ]

def tensor_to_memref2d(tensor: torch.Tensor) -> StridedMemRef2D:
    assert tensor.ndim == 2, "Tensor must be 2D"
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    ptr = tensor.data_ptr()
    sizes = (ctypes.c_int64 * 2)(tensor.shape[0], tensor.shape[1])
    strides = (ctypes.c_int64 * 2)(tensor.stride(0), tensor.stride(1))
    return StridedMemRef2D(ptr, ptr, 0, sizes, strides)

def tensor_to_memref1d(tensor: torch.Tensor) -> StridedMemRef1D:
    assert tensor.ndim == 1, "Tensor must be 1D"
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    ptr = tensor.data_ptr()
    sizes = (ctypes.c_int64 * 1)(tensor.shape[0])
    strides = (ctypes.c_int64 * 1)(tensor.stride(0))
    return StridedMemRef1D(ptr, ptr, 0, sizes, strides)

def unpack_and_reference_gemm(A: torch.Tensor, B_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Reference mathematical implementation of W4A16 GEMM in PyTorch
    """
    M, K = A.shape
    K_b, N_half = B_packed.shape
    N = N_half * 2
    
    # 1. Unpack uint8 packed bytes into INT4
    b_u8 = B_packed.to(torch.uint8)
    low_nibble = (b_u8 & 0x0F).to(torch.float32)
    high_nibble = ((b_u8 >> 4) & 0x0F).to(torch.float32)
    
    # Interleave low and high nibbles: even col = low, odd col = high
    W_int4 = torch.empty((K, N), dtype=torch.float32)
    W_int4[:, 0::2] = low_nibble
    W_int4[:, 1::2] = high_nibble
    
    # 2. Dequantize: (W - zero) * scale with Group Size = 128
    scales_expanded = scales.repeat_interleave(128, dim=0)[:K, :].to(torch.float32)
    zeros_expanded = zeros.repeat_interleave(128, dim=0)[:K, :].to(torch.float32)
    
    W_dequant = (W_int4 - zeros_expanded) * scales_expanded
    
    # 3. GEMM + Bias: C = A @ W + Bias (in native FP16)
    W_dequant_f16 = W_dequant.to(torch.float16)
    C_ref = torch.matmul(A, W_dequant_f16) + bias
    return C_ref

def test_shape(M, N, K, ciface_func):
    torch.manual_seed(42)
    assert K % 128 == 0, "K must be multiple of 128"
    assert N % 2 == 0, "N must be even"
    num_groups = K // 128
    
    A = torch.randn((M, K), dtype=torch.float16)
    B_packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8)
    B_packed_i8 = B_packed.view(torch.int8)
    scales = torch.rand((num_groups, N), dtype=torch.float16) * 0.1 + 0.01
    zeros = torch.randint(0, 8, (num_groups, N), dtype=torch.float16)
    bias = torch.randn(N, dtype=torch.float16)
    C_init = torch.zeros((M, N), dtype=torch.float16)
    
    memref_res = StridedMemRef2D(0, 0, 0, (ctypes.c_int64 * 2)(0, 0), (ctypes.c_int64 * 2)(0, 0))
    memref_A = tensor_to_memref2d(A)
    memref_B = tensor_to_memref2d(B_packed_i8)
    memref_scales = tensor_to_memref2d(scales)
    memref_zeros = tensor_to_memref2d(zeros)
    memref_bias = tensor_to_memref1d(bias)
    memref_C_init = tensor_to_memref2d(C_init)
    
    ciface_func(
        ctypes.byref(memref_res),
        ctypes.byref(memref_A),
        ctypes.byref(memref_B),
        ctypes.byref(memref_scales),
        ctypes.byref(memref_zeros),
        ctypes.byref(memref_bias),
        ctypes.byref(memref_C_init)
    )
    
    C_mlir = C_init
    C_ref = unpack_and_reference_gemm(A, B_packed, scales, zeros, bias)
    
    diff = (C_mlir.to(torch.float32) - C_ref.to(torch.float32)).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    
    act_f = C_mlir.to(torch.float32).view(-1)
    exp_f = C_ref.to(torch.float32).view(-1)
    cos_sim = torch.dot(act_f, exp_f).item() / (torch.norm(act_f).item() * torch.norm(exp_f).item() + 1e-12)
    
    is_close = torch.allclose(C_mlir, C_ref, rtol=1e-1, atol=2e-1)
    status = "✅ PASS" if is_close else "❌ FAIL"
    print(f"| Shape: M={M:<4} N={N:<4} K={K:<4} | CosSim: {cos_sim:.6f} | Max Diff: {max_err:.6f} | Mean Diff: {mean_err:.6f} | Status: {status} |")
    return is_close

def main():
    print("=" * 80)
    print("🧪 [Milestone 4] 启动 MLIR W4A16 算子多维度数值精度全景压测")
    print("=" * 80)
    
    dylib_path = os.path.join(os.path.dirname(__file__), "../kernels/libw4a16_mlir.dylib")
    if not os.path.exists(dylib_path):
        raise FileNotFoundError(f"Dylib not found: {dylib_path}")
    
    mlir_lib = ctypes.CDLL(dylib_path)
    ciface_func = getattr(mlir_lib, "_mlir_ciface_w4a16_gemm_fused")
    
    test_shapes = [
        (64, 64, 128),
        (64, 128, 128),
        (128, 64, 128),
        (128, 128, 128),
        (64, 64, 256),
        (128, 128, 256),
    ]
    
    all_passed = True
    for M, N, K in test_shapes:
        passed = test_shape(M, N, K, ciface_func)
        if not passed:
            all_passed = False
            
    print("=" * 80)
    if all_passed:
        print("🎉 [结论] 全维度测试用例 100% 全部通过！MLIR W4A16 算子数值正确性完全确立！")
    else:
        print("⚠️ [结论] 存在部分用例未通过，需进一步排查！")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
