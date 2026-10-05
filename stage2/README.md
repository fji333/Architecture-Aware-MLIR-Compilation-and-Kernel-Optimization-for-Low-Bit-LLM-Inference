# Project 2: Triton CPU (Apple M4) 后端解耦移植与 W4A16 低比特融合算子研发

本项目针对 **Apple Silicon (M4 / AArch64) 统一内存架构**，完整实现了基于 LLVM 19 / MLIR 编译基础设施的 **Triton AI 编译器后端解耦、纯 CPU JIT 引擎构建、6 层 MLIR 方言白盒降级分析、微架构感知瓦片调优、以及就地 Nibble 解包 W4A16 融合量化矩阵乘法算子开发**。

---

## 📂 项目工程标准目录架构

```text
Project2_Triton/
├── README.md                           # 📖 项目工程全局说明与快速上手指南
├── Project2_Research_Report.md         # 📑 顶会级学术研究报告 (含 10 大全量实测数据表与 5 张学术图表)
│
├── kernels/                            # 🚀 【核心 JIT 算子实现】
│   ├── fp16_gemm.py                    • 硬件感知优化版 FP16 原生矩阵乘法内核 (16x16 瓦片)
│   └── w4a16_gemm.py                   • 4-bit uint8 Nibble 就地解包与仿射融合算子 (零 DRAM 回写)
│
├── tests/                              # 🧪 【正确性与数值精度验证】
│   ├── test_correctness.py             • FP16 算子数值正确性测试套件
│   └── test_w4a16_correctness.py       • W4A16 与 PyTorch 逐元素反量化残差比对 (MAE=0.002193)
│
├── benchmarks/                         # ⏱️ 【性能压测与全量消融套件】
│   ├── benchmark_fp16.py               • FP16 矩阵乘法吞吐量基准压测
│   ├── benchmark_w4a16.py              • W4A16 端到端性能与加速比压测
│   └── run_massive_full_suite.py       • 包含 5 组核心消融维度的物理实测全量执行套件
│
├── analysis/                           # 🔬 【底层编译器 IR 提取与白盒诊断】
│   ├── dump_ir.py                      • 全自动提取 7 层 JIT 编译产物脚本
│   └── ir_dumps/                       • 提取保存的 7 层中间表示 (.mlir, .ll, .s, .so)
│
├── scripts/                            # 🛠️ 【环境构建与 C++ 补丁自动化脚本】
│   ├── build_triton_m4.sh              • 4 阶段全自动 LLVM 19 编译与 Triton CPU 构建脚本
│   └── patch_cpu.py                    • 自动化 C++ 源码热补丁工具
│
└── history/                            # 🏛️ 【完整历史演进归档与版本控制库】
    ├── README.md                       • 历史档案索引与代码迭代总表
    ├── 01_code_iterations/             • 13 个阶段源码历史快照 (从 C++ 解耦到 W4A16)
    ├── 02_terminal_logs/               • 7 个关键节点的原始终端物理执行日志
    ├── 03_ir_artifacts/                • 64x64 与 16x16 的 14 个编译器产物对比快照
    ├── 04_visual_charts/               • 5 张 300 DPI 学术级高清对比图表
    └── 05_performance_reports/         • 历史阶段性能对比表
```

---

## 🚀 快速上手与验证命令 (Quick Start)

### 1. 运行数值精度正确性验证
```bash
export TRITON_CPU_BACKEND=1

# 验证 FP16 原生算子
python tests/test_correctness.py

# 验证 W4A16 4-bit 量化融合算子
python tests/test_w4a16_correctness.py
```

### 2. 运行性能基准压测
```bash
# FP16 端到端压测 (对 PyTorch 保持 3x~7x 加速)
python benchmarks/benchmark_fp16.py

# W4A16 融合算子压测 (大矩阵取得最高 31.3x 加速)
python benchmarks/benchmark_w4a16.py

# 运行全量多维消融实验矩阵
python benchmarks/run_massive_full_suite.py
```

### 3. 提取编译器 7 层 IR 中间表示
```bash
python analysis/dump_ir.py
```

---

## 📊 核心研究产出速览

*   **核心学术报告**: [`Project2_Research_Report.md`](file:///Users/a15583507331/Desktop/AI_Compiler_PhD_Prep/Project2_Triton/Project2_Research_Report.md)
*   **完整历史归档**: [`history/`](file:///Users/a15583507331/Desktop/AI_Compiler_PhD_Prep/Project2_Triton/history/)
*   **端到端最大加速比**: **🚀 31.32x**（在真实大模型解码长条矩阵 $M=128, N=4096, K=4096$ 负载下，将耗时从 5.69 秒降至 0.18 秒，彻底打破内存墙）。
*   **汇编代码优化消减**: **88.7%**（通过微架构感知瓦片调优，将 ARM64 汇编指令从 206,586 行消减至 23,306 行，消除 L1 指令缓存污染）。
