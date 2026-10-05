#!/bin/bash
set -e

echo "=========================================="
echo " Triton CPU Backend (Apple M4) Build Script"
echo "=========================================="

echo "🚀 [1/4] Checking CMake and Ninja..."
if ! command -v cmake &> /dev/null; then brew install cmake; fi
if ! command -v ninja &> /dev/null; then brew install ninja; fi

echo "🚀 [2/4] Cloning LLVM Project..."
if [ ! -d "llvm-project" ]; then
    git clone --depth 1 https://github.com/llvm/llvm-project.git
fi
cd llvm-project
mkdir -p build && cd build

echo "🚀 [3/4] Compiling LLVM (AArch64 + MLIR)..."
echo "⚠️ 警告：这会跑满你 M4 所有的核心，预计耗时 30-50 分钟。风扇会狂转，请确保插上电源。"
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="AArch64" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
ninja
cd ../..

echo "🚀 [4/4] Cloning and Building Triton CPU Backend..."
if [ ! -d "triton" ]; then
    git clone https://github.com/triton-lang/triton.git
fi
cd triton

# 强制绑定本地编译的 LLVM，并开启 CPU 后端
export LLVM_SYSPATH=$(pwd)/../llvm-project/build
export TRITON_CPU_BACKEND=1

echo "📦 Installing Triton into active Conda environment..."
pip install -e python

echo "✅ 恭喜！Triton CPU 后端编译完成！"
