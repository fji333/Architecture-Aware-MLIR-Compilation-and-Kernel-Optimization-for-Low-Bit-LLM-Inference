#!/usr/bin/env python3
# ==============================================================================
# OmniSchedule: Systematic Multi-Tier Cache Tiling & Schedule Space Explorer
# Evaluates L1, L2, Register Tiling Combinations & Operator Fusion Strategies
# ==============================================================================

import os
import sys
import time
import json
import ctypes
import subprocess
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLVM_BIN = "/Users/a15583507331/Downloads/Project2_Triton/llvm-project/build/bin"
LLVM_LIB = "/Users/a15583507331/Downloads/Project2_Triton/llvm-project/build/lib"

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
    assert tensor.ndim == 2 and tensor.is_contiguous()
    ptr = tensor.data_ptr()
    sizes = (ctypes.c_int64 * 2)(tensor.shape[0], tensor.shape[1])
    strides = (ctypes.c_int64 * 2)(tensor.stride(0), tensor.stride(1))
    return StridedMemRef2D(ptr, ptr, 0, sizes, strides)

def tensor_to_memref1d(tensor: torch.Tensor) -> StridedMemRef1D:
    assert tensor.ndim == 1 and tensor.is_contiguous()
    ptr = tensor.data_ptr()
    sizes = (ctypes.c_int64 * 1)(tensor.shape[0])
    strides = (ctypes.c_int64 * 1)(tensor.stride(0))
    return StridedMemRef1D(ptr, ptr, 0, sizes, strides)

# ------------------------------------------------------------------------------
# 1. Schedule Generator & Dynamic JIT Compiler
# ------------------------------------------------------------------------------
def generate_schedule_mlir(config: dict, output_path: str):
    l2_tiles = config.get("l2_tile", None)
    l1_tiles = config["l1_tile"]
    reg_tiles = config["reg_tile"]
    fuse = config.get("fuse_producer", True)

    lines = [
        "module attributes {transform.with_named_sequence} {",
        "  transform.named_sequence @__transform_main(%root: !transform.any_op) {",
        '    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op',
        '    %dequant = transform.structured.match ops{["linalg.generic"]} attributes {filter_attr_name = "dequant"} in %root : (!transform.any_op) -> !transform.any_op',
    ]

    if l2_tiles:
        lines.append(f'    %tiled_matmul_l2, %loops_l2:2 = transform.structured.tile_using_for %matmul tile_sizes [{l2_tiles[0]}, {l2_tiles[1]}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)')
        if fuse:
            lines.append('    %fused_dequant, %fused_loop = transform.structured.fuse_into_containing_op %dequant into %loops_l2#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)')
        parent_op = "%tiled_matmul_l2"
    else:
        parent_op = "%matmul"

    lines.append(f'    %tiled_matmul_l1, %loops_l1:3 = transform.structured.tile_using_for {parent_op} tile_sizes [{l1_tiles[0]}, {l1_tiles[1]}, {l1_tiles[2]}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)')
    
    if not l2_tiles and fuse:
        lines.append('    %fused_dequant, %fused_loop = transform.structured.fuse_into_containing_op %dequant into %loops_l1#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)')

    lines.append(f'    %tiled_matmul_reg, %loops_reg:3 = transform.structured.tile_using_for %tiled_matmul_l1 tile_sizes [{reg_tiles[0]}, {reg_tiles[1]}, {reg_tiles[2]}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)')
    lines.append(f'    transform.structured.vectorize %tiled_matmul_reg vector_sizes [{reg_tiles[0]}, {reg_tiles[1]}, {reg_tiles[2]}] : !transform.any_op')
    lines.append('    transform.yield')
    lines.append('  }')
    lines.append('}')

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

def compile_config(config_id: str, schedule_path: str) -> str:
    tmp_dir = os.path.join(PROJECT_ROOT, f"kernels/tmp_{config_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    linalg_mlir = os.path.join(PROJECT_ROOT, "mlir_src/w4a16_linalg.mlir")
    scheduled_mlir = os.path.join(tmp_dir, "scheduled.mlir")
    bufferized_mlir = os.path.join(tmp_dir, "bufferized.mlir")
    llvm_mlir = os.path.join(tmp_dir, "llvm.mlir")
    ll_file = os.path.join(tmp_dir, "kernel.ll")
    obj_file = os.path.join(tmp_dir, "kernel.o")
    dylib_file = os.path.join(tmp_dir, "libkernel.dylib")

    # Pass 1: Transform
    cmd1 = f'"{LLVM_BIN}/mlir-opt" "{linalg_mlir}" --transform-preload-library="transform-library-paths={schedule_path}" --transform-interpreter -o "{scheduled_mlir}"'
    subprocess.run(cmd1, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Pass 2: One-shot Bufferization
    cmd2 = f'"{LLVM_BIN}/mlir-opt" "{scheduled_mlir}" --one-shot-bufferize="bufferize-function-boundaries" -o "{bufferized_mlir}"'
    subprocess.run(cmd2, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Pass 3: Lowering to LLVM Dialect
    cmd3 = f'"{LLVM_BIN}/mlir-opt" "{bufferized_mlir}" --convert-linalg-to-loops --expand-strided-metadata --lower-affine --lower-vector-multi-reduction --lower-vector-mask --convert-vector-to-scf --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --convert-ub-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o "{llvm_mlir}"'
    subprocess.run(cmd3, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Pass 4: MLIR to LLVM IR
    cmd4 = f'"{LLVM_BIN}/mlir-translate" --mlir-to-llvmir "{llvm_mlir}" -o "{ll_file}"'
    subprocess.run(cmd4, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Pass 5: LLC & Link
    cmd5 = f'"{LLVM_BIN}/llc" -O3 -filetype=obj -mcpu=apple-m4 "{ll_file}" -o "{obj_file}"'
    subprocess.run(cmd5, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cmd6 = f'clang++ -shared "{obj_file}" -L"{LLVM_LIB}" -Wl,-rpath,"{LLVM_LIB}" -lmlir_c_runner_utils -lmlir_runner_utils -o "{dylib_file}"'
    subprocess.run(cmd6, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    return dylib_file

# ------------------------------------------------------------------------------
# 2. Benchmark Runner across Evaluated Shapes
# ------------------------------------------------------------------------------
def evaluate_dylib(dylib_path: str, shapes: list, iters: int = 10) -> dict:
    lib = ctypes.CDLL(dylib_path)
    func = getattr(lib, "_mlir_ciface_w4a16_gemm_fused")

    results = {}
    for M, N, K in shapes:
        num_groups = K // 128
        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=torch.float16)
        B_packed = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8)
        scales = torch.rand((num_groups, N), dtype=torch.float16) * 0.1 + 0.01
        zeros = torch.randint(0, 8, (num_groups, N), dtype=torch.float16)
        bias = torch.randn(N, dtype=torch.float16) * 0.1
        C_out = torch.zeros((M, N), dtype=torch.float16)

        memref_res = StridedMemRef2D(0, 0, 0, (ctypes.c_int64 * 2)(0, 0), (ctypes.c_int64 * 2)(0, 0))
        memref_A = tensor_to_memref2d(A)
        memref_B = tensor_to_memref2d(B_packed.view(torch.int8))
        memref_scales = tensor_to_memref2d(scales)
        memref_zeros = tensor_to_memref2d(zeros)
        memref_bias = tensor_to_memref1d(bias)
        memref_C = tensor_to_memref2d(C_out)

        # Warmup
        for _ in range(3):
            func(ctypes.byref(memref_res), ctypes.byref(memref_A), ctypes.byref(memref_B),
                 ctypes.byref(memref_scales), ctypes.byref(memref_zeros), ctypes.byref(memref_bias), ctypes.byref(memref_C))

        # Benchmark Timing
        t0 = time.perf_counter()
        for _ in range(iters):
            func(ctypes.byref(memref_res), ctypes.byref(memref_A), ctypes.byref(memref_B),
                 ctypes.byref(memref_scales), ctypes.byref(memref_zeros), ctypes.byref(memref_bias), ctypes.byref(memref_C))
        t1 = time.perf_counter()

        avg_lat_ms = (t1 - t0) / iters * 1000.0
        gflops = (2.0 * M * N * K / (avg_lat_ms / 1000.0)) / 1e9
        
        # Verify Precision
        res_buf = (ctypes.c_uint16 * (M * N)).from_address(memref_res.aligned_ptr)
        act = torch.frombuffer(res_buf, dtype=torch.float16).reshape(M, N)
        
        b_u8 = B_packed.to(torch.uint8)
        W_int4 = torch.empty((K, N), dtype=torch.float32)
        W_int4[:, 0::2] = (b_u8 & 0x0F).to(torch.float32)
        W_int4[:, 1::2] = ((b_u8 >> 4) & 0x0F).to(torch.float32)
        s_exp = scales.repeat_interleave(128, dim=0)[:K, :].to(torch.float32)
        z_exp = zeros.repeat_interleave(128, dim=0)[:K, :].to(torch.float32)
        W_dequant = (W_int4 - z_exp) * s_exp
        exp = torch.matmul(A, W_dequant.to(torch.float16)) + bias

        act_f = act.to(torch.float32).view(-1)
        exp_f = exp.to(torch.float32).view(-1)
        cos_sim = torch.dot(act_f, exp_f).item() / (torch.norm(act_f).item() * torch.norm(exp_f).item() + 1e-12)
        mae = (act_f - exp_f).abs().mean().item()

        results[f"{M}x{N}x{K}"] = {
            "latency_ms": avg_lat_ms,
            "gflops": gflops,
            "cos_sim": cos_sim,
            "mae": mae
        }

    return results

# ------------------------------------------------------------------------------
# 3. Master Explorer Execution
# ------------------------------------------------------------------------------
def main():
    print("=" * 100, flush=True)
    print("🔬 [OmniSchedule] 启动 MLIR 多级缓存 (L1/L2/Register) 与调度优化空间全景探索器", flush=True)
    print("   Target Platform: Apple M4 (128KB L1D, 12MB L2, 4x 128-bit NEON Ports)", flush=True)
    print("=" * 100, flush=True)

    search_space = [
        {
            "id": "cfg_l1_conservative",
            "name": "1. 紧凑型 L1 缓存分块",
            "l2_tile": None,
            "l1_tile": [32, 32, 128],
            "reg_tile": [8, 8, 8],
            "fuse_producer": True,
            "desc": "以 32x32 小工作集常驻 L1D，低寄存器压力"
        },
        {
            "id": "cfg_l1_standard_fused",
            "name": "2. 标准 L1D 契合型分块 (M4 单层核心)",
            "l2_tile": None,
            "l1_tile": [64, 64, 128],
            "reg_tile": [8, 16, 8],
            "fuse_producer": True,
            "desc": "精准契合 M4 128KB L1D，NEON 16 列原生步进"
        },
        {
            "id": "cfg_l1_wide_vector",
            "name": "3. 宽向量高并发寄存器分块",
            "l2_tile": None,
            "l1_tile": [64, 128, 128],
            "reg_tile": [16, 16, 8],
            "fuse_producer": True,
            "desc": "展开 16 行并发累加，压榨 NEON 双发射算力"
        },
        {
            "id": "cfg_l2_hierarchical_std",
            "name": "4. 双层分层缓存调度 (L2+L1)",
            "l2_tile": [256, 512],
            "l1_tile": [64, 64, 128],
            "reg_tile": [8, 16, 8],
            "fuse_producer": True,
            "desc": "L2 (256x512) 锁存权重 + L1 (64x64) 二级分块"
        },
        {
            "id": "cfg_l2_deep_reduction",
            "name": "5. 深度规约分块 (K=256 跨 Group)",
            "l2_tile": [256, 512],
            "l1_tile": [64, 64, 256],
            "reg_tile": [8, 16, 8],
            "fuse_producer": True,
            "desc": "L1 单次容纳 2 个量化组，减少外层循环开销"
        },
        {
            "id": "cfg_l2_large_spatial",
            "name": "6. 空间扩展型大分块 (512x512)",
            "l2_tile": [512, 512],
            "l1_tile": [64, 128, 128],
            "reg_tile": [8, 16, 8],
            "fuse_producer": True,
            "desc": "面向超大矩阵，最大化 12MB L2 缓存局部性"
        },
    ]

    test_shapes = [
        (64, 64, 128),
        (64, 128, 128),
        (128, 64, 128),
        (128, 128, 128),
        (64, 64, 256),
        (128, 128, 256),
    ]

    exploration_records = []
    
    for item in search_space:
        cfg_id = item["id"]
        cfg_name = item["name"]
        print(f"\n⚙️ [探索方案: {cfg_name}]", flush=True)
        print(f"   L2 分块: {item['l2_tile']} | L1 分块: {item['l1_tile']} | 寄存器切片: {item['reg_tile']} | 算子融合: {item['fuse_producer']}", flush=True)
        print(f"   设计理念: {item['desc']}", flush=True)

        schedule_path = os.path.join(PROJECT_ROOT, f"mlir_src/schedules/generated_{cfg_id}.mlir")
        generate_schedule_mlir(item, schedule_path)

        t_compile_start = time.perf_counter()
        dylib_path = compile_config(cfg_id, schedule_path)
        compile_time_ms = (time.perf_counter() - t_compile_start) * 1000.0
        dylib_size_kb = os.path.getsize(dylib_path) / 1024.0

        eval_res = evaluate_dylib(dylib_path, test_shapes)
        
        # Calculate Average Throughput across shapes
        avg_gflops = sum(s["gflops"] for s in eval_res.values()) / len(eval_res)
        avg_cossim = sum(s["cos_sim"] for s in eval_res.values()) / len(eval_res)
        
        print(f"   => 编译耗时: {compile_time_ms:.1f} ms | 动态库体积: {dylib_size_kb:.1f} KB | 平均吞吐: {avg_gflops:.2f} GFLOPS | 平均余弦相似度: {avg_cossim:.6f}", flush=True)
        
        exploration_records.append({
            "config": item,
            "compile_time_ms": compile_time_ms,
            "dylib_size_kb": dylib_size_kb,
            "average_gflops": avg_gflops,
            "average_cos_sim": avg_cossim,
            "shape_details": eval_res
        })

    # Save Structured Telemetry
    artifacts_dir = os.path.join(PROJECT_ROOT, "benchmarks/artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    telemetry_path = os.path.join(artifacts_dir, "cache_exploration_telemetry.json")
    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump(exploration_records, f, indent=2, ensure_ascii=False)

    # Generate Markdown Summary Report
    md_path = os.path.join(artifacts_dir, "cache_hierarchy_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MLIR Transform Cache Hierarchy & Tiling Space Exploration Report\n\n")
        f.write("## 1. 探索方案与微架构对比矩阵\n\n")
        f.write("| 方案 ID | 方案名称 | L2 分块 | L1 分块 | 寄存器切片 | 平均吞吐 (GFLOPS) | 平均 CosSim | 编译开销 (ms) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rec in exploration_records:
            cfg = rec["config"]
            l2_str = str(cfg["l2_tile"]) if cfg["l2_tile"] else "None"
            f.write(f"| `{cfg['id']}` | **{cfg['name']}** | `{l2_str}` | `{cfg['l1_tile']}` | `{cfg['reg_tile']}` | **{rec['average_gflops']:.2f}** | {rec['average_cos_sim']:.6f} | {rec['compile_time_ms']:.1f} |\n")
        
        f.write("\n## 2. 各方案在细分矩阵形状下的吞吐对比 (GFLOPS)\n\n")
        f.write("| 方案名称 | 64x64x128 | 64x128x128 | 128x64x128 | 128x128x128 | 64x64x256 | 128x128x256 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rec in exploration_records:
            sd = rec["shape_details"]
            f.write(f"| **{rec['config']['name']}** | {sd['64x64x128']['gflops']:.2f} | {sd['64x128x128']['gflops']:.2f} | {sd['128x64x128']['gflops']:.2f} | {sd['128x128x128']['gflops']:.2f} | {sd['64x64x256']['gflops']:.2f} | {sd['128x128x256']['gflops']:.2f} |\n")

    print("\n" + "=" * 100, flush=True)
    print(f"📊 [探索完毕] 完整遥测数据已持久化至: {telemetry_path}", flush=True)
    print(f"📄 学术分析报告已生成至: {md_path}", flush=True)
    print("=" * 100, flush=True)

if __name__ == "__main__":
    main()
