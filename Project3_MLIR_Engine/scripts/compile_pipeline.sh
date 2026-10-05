#!/bin/bash
# ==============================================================================
# Project 3: End-to-End Automated Lowering Pipeline
# MLIR Linalg -> Transform Schedule -> Bufferization -> LLVM Dialect -> AArch64 Native .dylib
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_BIN="/Users/a15583507331/Downloads/Project2_Triton/llvm-project/build/bin"
LLVM_LIB="/Users/a15583507331/Downloads/Project2_Triton/llvm-project/build/lib"

echo "=============================================================================="
echo ">>> 启动 MLIR Transform 全自动编译与降级流水线 (Apple M4 AArch64) <<<"
echo "=============================================================================="

mkdir -p "${PROJECT_ROOT}/kernels"

echo "[Pass 1/5] 执行 Transform Dialect 调度 (多级分块 + 算子融合 + 向量化)..."
"${LLVM_BIN}/mlir-opt" \
  "${PROJECT_ROOT}/mlir_src/w4a16_linalg.mlir" \
  --transform-preload-library="transform-library-paths=${PROJECT_ROOT}/mlir_src/schedules/schedule_s6_deep_optimized.mlir" \
  --transform-interpreter \
  -o "${PROJECT_ROOT}/mlir_src/w4a16_scheduled.mlir"

echo "[Pass 2/5] 执行 One-Shot Bufferization (张量到内存引用映射)..."
"${LLVM_BIN}/mlir-opt" \
  "${PROJECT_ROOT}/mlir_src/w4a16_scheduled.mlir" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  -o "${PROJECT_ROOT}/mlir_src/w4a16_bufferized.mlir"

echo "[Pass 3/5] 执行渐进式降级 (Progressive Lowering to LLVM Dialect)..."
"${LLVM_BIN}/mlir-opt" \
  "${PROJECT_ROOT}/mlir_src/w4a16_bufferized.mlir" \
  --convert-linalg-to-loops \
  --expand-strided-metadata \
  --lower-affine \
  --lower-vector-multi-reduction \
  --lower-vector-mask \
  --convert-vector-to-scf \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-vector-to-llvm \
  --finalize-memref-to-llvm \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --convert-ub-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o "${PROJECT_ROOT}/mlir_src/w4a16_llvm_dialect.mlir"

echo "[Pass 4/5] 翻译为标准 LLVM IR 文本 (.ll)..."
"${LLVM_BIN}/mlir-translate" \
  --mlir-to-llvmir \
  "${PROJECT_ROOT}/mlir_src/w4a16_llvm_dialect.mlir" \
  -o "${PROJECT_ROOT}/kernels/w4a16.ll"

echo "[Pass 5/5] 使用 LLC 生成 Apple M4 原生目标机器码并链接为 .dylib..."
"${LLVM_BIN}/llc" -O3 -filetype=obj -mcpu=apple-m4 "${PROJECT_ROOT}/kernels/w4a16.ll" -o "${PROJECT_ROOT}/kernels/w4a16.o"
clang++ -shared "${PROJECT_ROOT}/kernels/w4a16.o" \
  -L"${LLVM_LIB}" \
  -Wl,-rpath,"${LLVM_LIB}" \
  -lmlir_c_runner_utils -lmlir_runner_utils \
  -o "${PROJECT_ROOT}/kernels/libw4a16_mlir.dylib"

echo "=============================================================================="
echo "✅ 编译流水线圆满完成！输出动态库: ${PROJECT_ROOT}/kernels/libw4a16_mlir.dylib"
echo "=============================================================================="
