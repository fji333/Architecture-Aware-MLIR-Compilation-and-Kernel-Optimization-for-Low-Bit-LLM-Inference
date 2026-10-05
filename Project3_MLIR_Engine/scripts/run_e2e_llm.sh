#!/bin/bash
# ==============================================================================
# OmniSchedule LLM Layer Patcher: HuggingFace Transformer Integration Demo
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="/opt/anaconda3/bin/python"

echo "=============================================================================="
echo "🧠 [OmniSchedule] 启动 HuggingFace Transformer 端到端大模型量化层拦截替换演示"
echo "=============================================================================="

# 1. 检查动态库是否存在，不存在则自动编译
if [ ! -f "${PROJECT_ROOT}/kernels/libw4a16_mlir.dylib" ]; then
    echo "⚠️ 核心动态库不存在，正在触发全自动编译流水线..."
    bash "${PROJECT_ROOT}/scripts/compile_pipeline.sh"
fi

# 2. 执行层拦截替换演示
PYTHONUNBUFFERED=1 "${PYTHON_EXEC}" "${PROJECT_ROOT}/engine/llm_layer_patcher.py"

echo "=============================================================================="
echo "✅ 端到端大模型算子拦截替换演示执行完毕！"
echo "=============================================================================="
