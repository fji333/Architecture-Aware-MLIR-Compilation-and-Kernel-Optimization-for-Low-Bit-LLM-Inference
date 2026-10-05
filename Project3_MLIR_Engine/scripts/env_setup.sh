#!/bin/bash
# Environment setup for Project 3 MLIR Engine

export LLVM_BUILD_DIR="/Users/a15583507331/Downloads/Project2_Triton/llvm-project/build"
export PATH="${LLVM_BUILD_DIR}/bin:${PATH}"
export PYTHON_EXEC="/opt/anaconda3/bin/python"
export TRITON_CPU_BACKEND="1"

echo "=== Project 3 MLIR Environment Configured ==="
echo "LLVM/MLIR Binaries: ${LLVM_BUILD_DIR}/bin"
echo "mlir-opt location: $(which mlir-opt || echo ${LLVM_BUILD_DIR}/bin/mlir-opt)"
echo "Python: ${PYTHON_EXEC}"
