# Project 3: MLIR Transform Dialect W4A16 Scheduling Engine & Edge LLM Acceleration

> **Apple M4 (ARMv9.2-A) 上的编译基础设施级 W4A16 GEMM 调度引擎与端侧大模型推理系统**
> *A Decoupled MLIR Transform Compilation Pipeline & High-Performance LLM Inference Framework on Apple Silicon*

---

## 1. 架构定位与核心突破 (Architecture & Contributions)

* **算法与调度彻底解耦 (Separation of Concerns)**:
  * **Payload IR (`mlir_src/w4a16_linalg.mlir`)**: 纯净声明非对称 4-bit 权重反量化与矩阵收缩乘法，不包含任何硬件特化细节。
  * **Transform Schedule (`mlir_src/schedules/`)**: 包含 6 套针对不同微架构优化假设的 MLIR Transform 调度脚本，涵盖多级缓存分块 (Tiling)、生产者-消费者跨循环原位融合 (Producer Fusion)、冗余向量访存外提 (Vector Transfer Hoisting)、内层规约循环展开 (Loop Unrolling) 与 128-bit NEON 原生向量指令映射。
* **端到端一键编译下沉流水线 (Progressive Lowering Pipeline)**:
  * `mlir-opt` $\to$ `One-Shot Bufferization` $\to$ `LLVM Dialect 渐进降级` $\to$ `mlir-translate` $\to$ `llc -mcpu=apple-m4` $\to$ 原生 `.dylib` 机器码。
* **极客级苛刻验证与全场景极限压测 (Geek-Level Verification & Stress Suite)**:
  * 涵盖 7 大类、共 47 项极客级测试用例（几何长宽比与单 Token 解码 GEMV、非 2 次幂奇数维度、前沿大模型投影、多分组深层累加、12 种极端对抗 Fuzzing、多调度交叉一致性与 1000 轮长稳内存泄漏审计），**综合通过率 100.0% (Bit-Exact Verified)**，余弦相似度 $\ge 0.999995$。

---

## 2. 规范化目录架构一览 (Clean & Logical Directory Structure)

```text
Project3_MLIR_Engine/
├── README.md                      # 系统总览、编译流水线与快速上手指南
├── docs/                          # 中英文学术级技术研究报告 (IEEE/ACM 格式)
│   ├── Project3_Research_Report.md
│   └── Project3_Research_Report_EN.md
├── mlir_src/                      # MLIR 核心算法表达与 Transform 调度脚本库
│   ├── w4a16_linalg.mlir          # 算法声明式 Linalg Generic IR (Payload)
│   ├── schedule_m4.mlir           # 默认 Apple M4 调度脚本
│   ├── w4a16_scheduled.mlir       # 调度应用后 IR 快照
│   ├── w4a16_bufferized.mlir      # 内存分配与 One-Shot Bufferize 后 IR
│   ├── w4a16_llvm_dialect.mlir    # 渐进降级为 LLVM Dialect IR
│   └── schedules/                 # 6 套深度 Transform 调度优化脚本库
│       ├── schedule_s1_no_fusion.mlir          # S1: 无融合基准 (DRAM 往返)
│       ├── schedule_s2_l1_only.mlir            # S2: 1-Tier L1 缓存分块 + 生产者原位融合
│       ├── schedule_s3_l2_l1_hierarchical.mlir # S3: 2-Tier 多级分块 (L2+L1) + 权重锁存
│       ├── schedule_s4_m4_specialized.mlir     # S4: 3-Tier 紧凑分块 (L1+Reg+NEON 向量化)
│       ├── schedule_s5_wide_neon_specialized.mlir # S5: 宽向量 [16,16,8] 双 SIMD 累加器
│       └── schedule_s6_deep_optimized.mlir     # S6: 顶级 SOTA 调度 (Transfer Hoisting + 2x Unroll)
├── kernels/                       # 编译产物与原生目标代码
│   ├── w4a16.ll                   # 标准 LLVM IR 文本
│   ├── w4a16.o                    # AArch64 原生目标机器码
│   └── libw4a16_mlir.dylib        # 导出的 C-ABI 动态共享库
├── engine/                        # 大模型端到端推理集成
│   └── llm_layer_patcher.py       # HuggingFace Transformer 线性层拦截替换模块
├── tests/                         # 单元测试与极客级 47 项全场景测试套件
│   ├── test_mlir_w4a16_correctness.py  # 基础多维度正确性单测 (100% PASS)
│   ├── comprehensive_test_suite.py     # 7 大类别 47 项全场景苛刻压测套件 (100% PASS)
│   └── artifacts/                 # 结构化测试遥测与分析报告
│       ├── test_telemetry.json         # 47 项测试完整遥测 JSON 数据
│       └── test_verification_report.md # 学术级测试验证报告
├── benchmarks/                    # 物理硬件性能压测与多级缓存空间探索
│   ├── benchmark_w4a16_mlir.py    # 算力吞吐与延迟全景评测
│   ├── cache_tile_space_explorer.py # L1/L2 缓存与分块参数空间自动搜索器
│   └── artifacts/                 # 缓存探索实验报告与遥测数据
│       ├── cache_exploration_telemetry.json # 缓存探索原始遥测数据
│       └── cache_hierarchy_analysis.md      # 多级缓存微架构分析报告
└── scripts/                       # 规范化一键执行脚本 (全带自检与容错)
    ├── env_setup.sh               # 环境变量配置
    ├── compile_pipeline.sh        # 一键执行 MLIR 端到端全自动编译流水线
    ├── run_tests.sh               # 一键执行全量 47 项测试套件 (包含单测与极限压测)
    ├── run_benchmarks.sh          # 一键执行物理硬件算力实测
    ├── run_cache_explorer.sh      # 一键执行 L1/L2 缓存与调度空间探索
    └── run_e2e_llm.sh             # 一键执行大模型层拦截演示
```

---

## 3. 规范化执行命令一览 (Quick Start Commands)

所有脚本均位于 `scripts/` 目录下，命名与逻辑严密对称，内置动态库自检与自动容错：

### 1. 配置基础环境

```bash
source scripts/env_setup.sh
```

### 2. 执行端到端 MLIR 编译流水线 (Pass 1~5)

```bash
./scripts/compile_pipeline.sh
```

*自动完成 MLIR Transform 调度、One-Shot Bufferization、渐进式降级，并在 `kernels/` 下生成 Apple M4 原生动态库 `libw4a16_mlir.dylib`。*

### 3. 一键运行全量 47 项极客级测试套件 (47 Test Cases, 100% PASS)

```bash
./scripts/run_tests.sh
```

*执行 Stage 1 基础单元测试与 Stage 2 极客级 7 大类别全景压测（含长宽比/GEMV、非2次幂、LLM投影、分组深层累加、对抗Fuzzing、多调度交叉一致性与 1000 轮长稳），遥测数据自动存入 `tests/artifacts/`。*

### 4. 一键执行物理硬件算力基准测试 (Apple M4 Benchmark)

```bash
./scripts/run_benchmarks.sh
```

### 5. 一键执行 L1/L2 多级缓存与调度空间自动化探索 (Cache Hierarchy Explorer)

```bash
./scripts/run_cache_explorer.sh
```

*对比 6 种典型微架构缓存分块与算子融合策略，分析报告自动生成至 `benchmarks/artifacts/cache_hierarchy_analysis.md`。*

### 6. 一键执行端到端大模型线性层拦截与加速演示

```bash
./scripts/run_e2e_llm.sh
```

---

## 4. 核心学术报告与技术文档

项目完整的研究动机、微架构理论推导、Roofline 模型分析与实验数据请查阅 `docs/` 目录：

- **中文学术报告**: [`docs/Project3_Research_Report.md`](file:///Users/a15583507331/Downloads/Project3_MLIR_Engine/docs/Project3_Research_Report.md)
- **英文学术报告**: [`docs/Project3_Research_Report_EN.md`](file:///Users/a15583507331/Downloads/Project3_MLIR_Engine/docs/Project3_Research_Report_EN.md)
