#!/usr/bin/env python3
import ctypes
import os
import sys
import time
import torch

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
    ptr = tensor.data_ptr()
    sizes = (ctypes.c_int64 * 2)(tensor.shape[0], tensor.shape[1])
    strides = (ctypes.c_int64 * 2)(tensor.stride(0), tensor.stride(1))
    return StridedMemRef2D(ptr, ptr, 0, sizes, strides)

def tensor_to_memref1d(tensor: torch.Tensor) -> StridedMemRef1D:
    ptr = tensor.data_ptr()
    sizes = (ctypes.c_int64 * 1)(tensor.shape[0])
    strides = (ctypes.c_int64 * 1)(tensor.stride(0))
    return StridedMemRef1D(ptr, ptr, 0, sizes, strides)

def benchmark_single_shape(M, N, K, ciface_func, iters=3):
    A = torch.randn((M, K), dtype=torch.float16)
    B_packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8)
    B_packed_i8 = B_packed.view(torch.int8)
    num_groups = max(1, K // 128)
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
    
    # Warmup
    ciface_func(
        ctypes.byref(memref_res), ctypes.byref(memref_A), ctypes.byref(memref_B),
        ctypes.byref(memref_scales), ctypes.byref(memref_zeros), ctypes.byref(memref_bias), ctypes.byref(memref_C_init)
    )
    
    t0 = time.perf_counter()
    for _ in range(iters):
        ciface_func(
            ctypes.byref(memref_res), ctypes.byref(memref_A), ctypes.byref(memref_B),
            ctypes.byref(memref_scales), ctypes.byref(memref_zeros), ctypes.byref(memref_bias), ctypes.byref(memref_C_init)
        )
    t1 = time.perf_counter()
    
    avg_latency_ms = (t1 - t0) / iters * 1000.0
    total_flops = 2.0 * M * N * K
    gflops = (total_flops / (avg_latency_ms / 1000.0)) / 1e9
    
    print(f"| {M:<5} x {N:<5} x {K:<5} | MLIR Latency: {avg_latency_ms:>10.2f} ms | MLIR Throughput: {gflops:>8.2f} GFLOPS | Total Compute: {total_flops/1e9:>6.2f} GFLOPs |", flush=True)
    return avg_latency_ms, gflops

def main():
    dylib_path = os.path.join(os.path.dirname(__file__), "../kernels/libw4a16_mlir.dylib")
    lib = ctypes.CDLL(dylib_path)
    ciface_func = lib._mlir_ciface_w4a16_gemm_fused
    
    print("=" * 110)
    print("🚀 [Ultra-Large Scale Benchmark] 超大矩阵规模 (Huge Matrix Shapes) 实测")
    print("=" * 110)
    
    shapes = [
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (2048, 2048, 2048),
        (128, 8192, 8192),
        (256, 8192, 8192),
    ]
    
    for M, N, K in shapes:
        benchmark_single_shape(M, N, K, ciface_func, iters=3)
        
    print("=" * 110)

if __name__ == "__main__":
    main()
