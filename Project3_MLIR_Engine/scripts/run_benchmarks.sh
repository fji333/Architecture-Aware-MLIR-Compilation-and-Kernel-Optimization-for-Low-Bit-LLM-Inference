#!/bin/bash
# ==============================================================================
# OmniSchedule Benchmark Runner: Hardware Throughput & Latency Evaluation
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="/opt/anaconda3/bin/python"

echo "=============================================================================="
echo "🚀 [OmniSchedule] 启动 Apple M4 物理硬件基准测速流水线"
echo "=============================================================================="

# 1. 检查动态库是否存在，不存在则自动编译
if [ ! -f "${PROJECT_ROOT}/kernels/libw4a16_mlir.dylib" ]; then
    echo "⚠️ 核心动态库不存在，正在触发全自动编译流水线..."
    bash "${PROJECT_ROOT}/scripts/compile_pipeline.sh"
fi

# 2. 执行算力基准测试
PYTHONUNBUFFERED=1 "${PYTHON_EXEC}" "${PROJECT_ROOT}/benchmarks/benchmark_w4a16_mlir.py"

echo "=============================================================================="
echo "✅ 算力测速执行完毕！"
echo "=============================================================================="
