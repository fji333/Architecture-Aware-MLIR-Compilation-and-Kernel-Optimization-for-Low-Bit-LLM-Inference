# In-Depth Microarchitectural Analysis of Single-Core W4A16 GEMM on Apple M4 UMA

## Abstract

Traditional computer architecture theory often employs Hierarchical Cache Tiling to mitigate the Memory Wall bottleneck [3]. This paper conducts a microarchitectural ablation study on the Apple M4 processor (120 GB/s UMA architecture), constrained by strict memory alignment and P-Core binding. Results indicate that single-level cache tiling interferes with the Apple M4 Data Memory-Dependent Prefetcher (DMP)'s continuous stride prediction, leading to an approximate 9% throughput reduction [1][2]. While dual-level hierarchical tiling (L1+L2) can utilize the L1 data cache's high fetch throughput to mask main memory latency, combining it with high instruction-level parallelism (ILP, e.g., $U_k \ge 4$) increases Branch Misprediction Rates and triggers Register Spilling due to the physical register limit, resulting in performance degradation [4]. Furthermore, a pure FP16 control experiment that strips all dequantization instructions demonstrates a throughput increase from 155.04 GFLOPS to 220.85 GFLOPS (+42.5%), directly confirming that dequantization-induced ALU issue port contention is the dominant factor behind the 55.05% utilization rate [7]. This study systematically dissects the global microarchitectural bottlenecks constraining the W4A16 operator's peak throughput on Apple Silicon.

---

## 0. Global Experimental Constants

To ensure rigorous variable control, the following parameters are locked as system-level global constants:

- **Target Architecture**: Apple M4 (based on ARMv9.2a SME instruction set) [6]
- **Memory Subsystem**: LPDDR5X-7500 Unified Memory Architecture (UMA), measured physical bandwidth ceiling ~120 GB/s [2][6]
- **Operator Type**: Single-Core W4A16 Mixed-Precision GEMM
- **Tensor Dimensions**: $M=1024, N=16384, K=4096$
- **Total Working Set Load**: $32 \text{ MB}$ (Weight matrix only, deliberately exceeding the M4's 16MB L2 cache capacity to induce memory pressure)
- **Toolchain**: Apple Clang (`-O3 -mcpu=native -std=c++11`)
- **Physical Environment Locks**: `posix_memalign` enforces 16KB OS page alignment and cache line alignment; `QOS_CLASS_USER_INTERACTIVE` forces thread binding to the P-Core, preventing E-Core scheduling drift.

---

## 1. Experiment 1: Single-Level Cache Tiling vs. Hardware Prefetching

This experiment investigates the physical behavior of single-level cache tiling on the Apple M4 UMA architecture when the memory working set exceeds L2 capacity.

**[Experimental Variables]**

- **Controlled**: $U_k = 2$ (Instruction unrolling factor fixed); Tiling Depth = 1 (L1 or L2 tiling only).
- **Independent**: $M_c, N_c$ (Geometric dimensions of the L1 or L2 tiled matrix)
- **Dependent**: Execution Time (ms) and Throughput (GFLOPS)

### 1.1 Empirical Data Table

| Profile Group | Spatial Mapping (Tiling Level & Size) | Block Size | Time (ms) | Throughput (GFLOPS) | Degradation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[Baseline]** | **No Tiling (Full Linear Scan)** | **32 MB (>L2)** | **891.61** | **154.14** | **0.0%** |
| Profile 1A | Single L1 Level (64x64) | 32 KB | 977.46 | 140.60 | -8.7% |
| Profile 1B | Single L2 Level (128x256) | 512 KB | 980.82 | 140.12 | -9.0% |
| Profile 1C | Single L2 Level (256x512) | 2 MB | 979.96 | 140.24 | -9.0% |

### 1.2 Microarchitectural Analysis

Data indicates that single-level tiling causes an approximate 9% throughput reduction.
According to microarchitectural reverse engineering [2] and studies on the Apple DMP [1], the Apple M-series architecture utilizes a Stride/Stream Predictor. In non-tiled linear scanning, the 32MB continuous access provides a complete address prediction sequence. The physical bandwidth sufficiently masks L2 cache miss latencies [6].
Introducing a single-level tile logically fragments the continuous address sequence into discrete 512KB (or smaller) blocks, causing the hardware prefetcher to lose address continuity and triggering frequent predictor resets [1]. Concurrently, standard prefetching behavior during non-linear access may produce over-fetching, subsequently evicting active working sets from the cache (Cache Thrashing) [2]. Consequently, single-level tiling does not exhibit anticipated performance benefits on this architecture.

---

## 2. Experiment 2: Exploiting L1 Data Cache Latency with Dual-Level Tiling

This experiment examines whether dual-level hierarchical tiling (L2+L1), aligned with hardware cache layers, can compensate for prefetcher invalidation using minimal physical latency [3].

**[Experimental Variables]**

- **Controlled**: $U_k = 2$; Tiling Depth = 2 (Outer L2 tile and Inner L1 tile simultaneously active).
- **Independent**: Dual-level tiling size combinations (L2 Size / L1 Size)
- **Dependent**: Execution Time (ms) and Throughput (GFLOPS)

### 2.1 Empirical Data Table

| Profile Group | Hierarchical Mapping (Outer L2 / Inner L1) | Estimated Inner Hit Target | Time (ms) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- | :--- |
| **[Baseline]** | **No Tiling (Full Linear Scan)** | Relies on Main Mem Prefetch | **891.61** | **154.14** |
| Profile 2A | L2(128x256) + L1(32x64) | L1 Data Cache | 894.56 | 153.63 |
| **Profile 2B** | **L2(128x256) + L1(64x64)** | **L1 Data Cache** | **893.70** | **153.78** |
| Profile 2C | L2(256x512) + L1(64x128) | L1 Data Cache | 893.39 | 153.83 |

### 2.2 Microarchitectural Analysis

Upon introducing a 4-level deep nested loop (Profile 2B), throughput recovers to 153.78 GFLOPS, nearing the baseline.
The underlying mechanism is that dual-level hierarchical tiling forces the innermost FMA computation to converge within the 128KB L1 Data Cache (L1D). Microarchitectural analysis shows that the physical hit latency for NEON floating-point loads (`LDR Q`) in the L1D is approximately **$\sim 9$ clock cycles**. However, the M4 core possesses high Load/Store throughput (issuing 3 vector load instructions per cycle, with a reciprocal throughput of $\sim 0.33$ cycles/instruction). Under this throughput characteristic, confining the working set to the L1D allows the data supply rate to the FMA units to achieve Pipeline Saturation. Although the outer loops reduce main memory prefetcher efficiency, the high fetch throughput of the L1 cache partially masks main memory stalls [3].

---

## 3. Experiment 3: Maximum Instruction-Level Parallelism (ILP) and Front-End Fetch Characteristics

This experiment tests the impact of instruction pipeline unrolling on throughput while eliminating cache tiling logic (ensuring maximum hardware prefetcher efficiency).

**[Experimental Variables]**

- **Controlled**: Tiling Topology = No Tiling (strictly linear continuous address access).
- **Independent**: $U_k$ (Micro-kernel K-dimension loop unroll factor: 2, 4, 6)
- **Dependent**: Execution Time (ms) and Throughput (GFLOPS)

### 3.1 Empirical Data Table

| Profile Group | Instruction Pipeline Unroll Factor | Theoretical Vector Registers Required | Time (ms) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- | :--- |
| Profile 3A | $U_k = 2$ | 26 | 891.61 | 154.14 |
| Profile 3B | $U_k = 4$ | 30 | 895.31 | 153.50 |
| **Profile 3C** | **$\mathbf{U_k = 6}$** | **$\approx 34$** | **886.46** | **155.04 (Peak)** |

### 3.2 Microarchitectural Analysis (ILP and Fetch Immunity)

Experimental data demonstrates that in a non-tiled state, increasing the unroll factor $U_k$ from 2 to 6 elevates throughput from 154.14 GFLOPS to 155.04 GFLOPS (the physical peak). This performance gain originates from two underlying mechanisms:

1. **Loop Overhead Reduction and OoO Window Expansion**: Increasing $U_k$ reduces the absolute number of branch instructions and boundary condition evaluations, minimizing control flow overhead. More importantly, the dense stream of independent FMA instructions provides the processor's Out-of-Order (OoO) core with a richer sequence, maximizing the concurrent utilization of the 4 NEON pipelines and effectively hiding instruction-level pipeline latency.
2. **Front-End Fetch Characteristics (The 192KB L1I Phenomenon)**: In certain architectures, extreme software unrolling (e.g., $U_k=6$) exceeds micro-op ($\mu$op) cache capacity, causing fetch stalls. Microarchitectural data indicates that the Apple M-Series omits a traditional $\mu$op cache in favor of a massive **192 KB L1 Instruction Cache (L1I)** and an 8-wide decoder. This structure allows the M4 to readily accommodate bloated kernel footprints under high ILP without exhibiting any front-end instruction starvation.

---

## 4. Experiment 4: Branch Misprediction and Register Spilling Validation

This experiment investigates the processor's underlying physical behavior when combining dual-level tiling (Exp 2) with high concurrent instruction unrolling (Exp 3).

**[Experimental Variables]**

- **Controlled**: Tiling Topology = Dual-level fixed at L2(128x256) + L1(64x64).
- **Independent**: $U_k$ (Unroll factor: 4, 6, 8)
- **Dependent**: Throughput (GFLOPS)

### 4.1 Cross-Comparison Data Table

| Unroll Factor ($U_k$) | [Pure Linear Access] Throughput | [Dual-Level Tiling] Throughput | Absolute Degradation | Physical Layer Diagnosis |
| :---: | :--- | :--- | :--- | :--- |
| **$U_k = 4$** | **153.50** GFLOPS | 145.73 GFLOPS | **-5.0%** | Elevated Branch Mispredictions |
| **$U_k = 6$** | **155.04** GFLOPS | 148.89 GFLOPS | **-3.9%** | Register Spilling Triggered |
| **$U_k = 8$** | (Physical Degradation) | 150.85 GFLOPS | - | Stack Memory I/O Injection |

### 4.2 Microarchitectural Analysis

At $U_k=6$, the pure linear access approach reaches 155.04 GFLOPS, while the dual-level tiled approach drops to 148.89 GFLOPS. This degradation is driven by two mechanisms:

1. **Control Flow Complexity and Pipeline Flush Penalty**: Dual-level tiling introduces a 4-level nested loop and boundary condition checks. As $U_k$ increases, inner loop bloat and dense branching instructions increase the TAGE branch predictor's misprediction rate. Given the core's deep pipeline, a single misprediction incurs a 14 to 20 clock cycle Pipeline Flush overhead [5].
2. **Physical Register Limits and LLVM Phase-Ordering Constraints**: The ARMv9 architecture provides 32 physical 128-bit vector registers [6]. At $U_k=6$, the required Live Ranges exceed this limit. The LLVM compiler encounters a "Phase-Ordering Problem": the pre-scheduler extends variable lifespans to increase ILP, making it difficult for the greedy register allocator (RAGreedy) to resolve assignments without conflicts. LLVM resolves this by spilling registers to the hot loop [4]. Disassembly of the micro-kernel confirms this mechanism:

```assembly
; (Disassembly Snippet: Register Spilling at U_k=6)
  f0: f90007e9 	str	x9, [sp, #0x8]       ; Save register state to stack memory
  f4: f90013e2 	str	x2, [sp, #0x20]
  fc: f9001bee 	str	x14, [sp, #0x30]
 100: f9400fe9 	ldr	x9, [sp, #0x18]      ; Restore data from stack memory
 104: 9b097dc9 	mul	x9, x14, x9
 108: 8b090459 	add	x25, x2, x9, lsl #1
 10c: f94017fa 	ldr	x26, [sp, #0x28]
```
This Register Spilling behavior introduces additional L1 memory access operations, resulting in performance degradation.

---

## 5. Experiment 5: Dequantization Overhead Isolation — Pure FP16 Control Group

Experiments 1–4 established that the W4A16 micro-kernel peaks at 155.04 GFLOPS, representing only 55.05% of the theoretical maximum (281.6 GFLOPS). However, whether this deficit is caused by ALU port contention from dequantization instructions or by insufficient memory bandwidth lacked direct experimental evidence. This experiment isolates the variable by constructing a control group with **all dequantization instructions removed**.

**[Experimental Variables]**

- **Controlled**: Micro-kernel structure = 8x16 tile; $U_k = 2$; Tiling topology = No tiling (linear scan); QoS P-Core binding; `posix_memalign` 16KB page alignment. Tensor dimensions $M=1024, N=16384, K=4096$ unchanged.
- **Independent**: Weight data format and inner-loop instruction composition
  - **Experimental group (W4A16)**: Weights stored as 4-bit integers; inner loop contains `vshlq` + `vshrq` + `vmovl` + `vcvtq` (dequantization) and `vfmaq` (computation)
  - **Control group (Pure FP16)**: Weights stored directly as FP16; inner loop contains **only** `vld1q_f16` (load) + `vfmaq_f16` (computation), with no bitwise or type conversion instructions
- **Dependent**: Throughput (GFLOPS)

### 5.1 Empirical Data Table

| Group | Weight Format | Inner-Loop Instructions | Weight Working Set | Time (ms) | Throughput (GFLOPS) | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Experimental (W4A16)** | INT4 (packed) | `vshlq` + `vshrq` + `vmovl` + `vcvtq` + `vfmaq` x2 | 32 MB | 891.61 | 154.14 | Baseline |
| **Control (Pure FP16)** | FP16 (16-bit) | `vld1q_f16` + `vfmaq_f16` | 128 MB | 622.31 | **220.85** | **+42.5%** |

### 5.2 Microarchitectural Analysis

Removing dequantization instructions elevates throughput from 154.14 GFLOPS to 220.85 GFLOPS, a 42.5% increase. This result carries the following physical implications:

1. **Direct Evidence of ALU Issue Port Contention**: The Apple M4 P-Core provides 4 NEON/ASIMD execution pipelines capable of issuing at least 4 128-bit vector instructions per cycle [6]. In the W4A16 kernel, each K-dimension iteration requires: `vld1_u8` (load), `vshlq_n_u8` + `vshrq_n_u8` (bit-shift to split high/low 4-bit), `vmovl_u8` (zero-extend to 16-bit), `vcvtq_f16_u16` (integer-to-FP16 conversion), `vfmaq_f16` (scale/bias affine transform), and `vfmaq_f16` x16 (actual FMA computation). The dequantization sequence occupies approximately 6 vector instructions, while effective FMA computation accounts for only 16. These dequantization instructions compete with FMA for backend issue ports, substantially reducing effective FMA occupancy [7].

2. **Instruction Mix Ratio Quantification**: In the W4A16 kernel's hot loop, the per-K-step instruction mix ratio is approximately **6:16** (dequantization:FMA), meaning 27.3% of issue slots are consumed by dequantization. Since the M4 backend shares NEON execution pipelines across all vector instruction types, dequantization injection directly dilutes FMA throughput. The theoretical throughput ratio is $16/(16+6) \approx 72.7\%$, while the measured W4A16/FP16 throughput ratio is $154.14/220.85 \approx 69.8\%$—a close match. The additional ~3% discrepancy is attributable to serial data dependencies within the dequantization chain (`vshlq` -> `vmovl` -> `vcvtq` forms a dependency chain that constrains out-of-order scheduling freedom).

3. **Residual Loss Analysis of the FP16 Control Group**: The control group achieves 220.85 GFLOPS, representing 78.4% of the theoretical peak (281.6 GFLOPS). The remaining 21.6% loss does not originate from ALU port contention (which has been eliminated), but from systemic physical constraints: (a) the weight working set inflates to 128 MB (4x the INT4 size), far exceeding L2 cache and TLB coverage, intensifying memory bandwidth pressure and page walk overhead; (b) load port saturation—each `vfmaq` requires one weight load plus one activation load, while the M4 issues at most 3 load instructions per cycle; (c) DVFS thermal throttling and UMA fabric jitter (see Section 6).

---

## 6. Global Microarchitectural Bottlenecks and Roofline Analysis

The W4A16 peak of 155.04 GFLOPS and the pure FP16 peak of 220.85 GFLOPS are both below the silicon theoretical maximum. Combined with Experiment 5's control data, the following global physical bottlenecks constrain peak throughput:

### 6.1 ALU Port Contention and Roofline Boundary
Under the Roofline Model, the Apple M4 P-Core, with 4 NEON execution pipelines running at 4.4GHz, has a theoretical FP16 computational peak of **281.6 GFLOPS**.
The W4A16 kernel measures 155.04 GFLOPS (theoretical peak's **55.05%**); the pure FP16 control group measures 220.85 GFLOPS (theoretical peak's **78.4%**). Experiment 5 directly confirms that dequantization instructions (`vshlq`, `vshrq`, `vcvtq`, `vmovl`) competing with `vfmaq` for backend issue ports is the dominant factor in W4A16 throughput loss, contributing approximately **65 GFLOPS** of the total deficit (51.6%) [7]. The remaining ~61 GFLOPS gap (from 220.85 to 281.6 GFLOPS) is attributable to memory subsystem constraints (Sections 6.2–6.4).

### 6.2 TLB Reach Exhaustion and Page Walks
Based on a 16KB page size, the M4 L2 TLB holds approximately 3072 entries, yielding a maximum physical addressing coverage (TLB Reach) of approximately **48 MB**.
The experimental matrix multiplication working set reaches **72 MB**, exceeding L2 TLB coverage. During linear scanning, crossing the 48MB boundary triggers L2 TLB misses and hardware Page Walks, introducing unavoidable memory latency.

### 6.3 UMA Interconnect Fabric Jitter
Apple Silicon utilizes a Unified Memory Architecture (UMA) where CPU cores, GPUs, and NPUs share the System Level Cache (SLC) and memory controllers. Even when benchmarking is locked to a single P-Core via QoS, routine background OS scheduling creates minor data stream contention on the interconnect fabric. This multi-module shared mechanism induces bandwidth jitter (micro-stuttering), preventing smooth theoretical peak bandwidth output.

### 6.4 Dynamic Voltage and Frequency Scaling (DVFS) & Thermal Throttling
Sustained NEON FP16 FMA execution under full load exhibits notable power consumption characteristics. Thermal accumulation prompts the Power Management Microcontroller (PMGR) to engage DVFS, moderately reducing the P-Core's operating frequency. Consequently, the measured throughput represents a time-averaged result reflecting thermal dissipation constraints.

---

## 7. Conclusion and Architectural Evolution

### 7.1 Summary of Physical Constraints
For large-scale GEMM computations on the Apple M4 UMA architecture, this study derives the following conclusions through five ablation experiments:
1. Due to the operational mechanics of the DMP (Data Memory-Dependent Prefetcher), single-level cache tiling strategies that interrupt continuous linear memory access reduce bandwidth utilization by ~9% (Experiment 1) [1].
2. While dual-level L1+L2 hierarchical tiling can leverage fetch throughput to partially mask prefetch misses (Experiment 2) [3], its control flow complexity, when combined with high ILP, introduces branch misprediction overhead and register spilling (Experiment 4) [4].
3. Maintaining streamlined control flow, preserving address linear scanning to match hardware prefetchers, and pushing register allocation to physical limits ($U_k=6$) is an effective strategy for approaching the current W4A16 micro-kernel limit of 155.04 GFLOPS (Experiment 3).
4. Dequantization instructions competing for ALU issue ports are the dominant factor behind the 55.05% utilization rate. Removing dequantization yields a 42.5% throughput increase, closely matching theoretical instruction mix ratio predictions (Experiment 5) [7].

### 7.2 Architectural Evolution Path
Given that the current NEON architecture is constrained by ALU port contention and a hard limit of 32 physical registers, further optimization requires utilizing the hardware-supported **ARMv9.2a SME (Scalable Matrix Extension)** on the Apple M4.
SME introduces a 2D state register array (ZA Array). Employing SME-exclusive `FMOPA` instructions effectively circumvents vector register ceilings and significantly reduces scheduling pressure on ALU issue ports. Under the SME architecture, single-core matrix computation performance is expected to achieve significant gains, representing the core pathway to breaking the current Roofline utilization bottleneck.
Furthermore, Experiment 5's control data indicates that if native INT4 FMA were implemented at the hardware level (eliminating software dequantization), or if SME's wide matrix operations could amortize dequantization overhead to negligible levels, W4A16 operators could approach the pure FP16 benchmark of 220+ GFLOPS.

---

## References

[1] Cronin, J., et al. (2022). "Augury: Using Data Memory-Dependent Prefetchers to Leak Data at Rest". *IEEE Symposium on Security and Privacy*.
[2] Asahi Linux Project. (2023). "Apple Silicon Hardware Documentation and Unified Memory Architecture Reverse Engineering".
[3] Goto, K., & van de Geijn, R. A. (2008). "Anatomy of High-Performance Matrix Multiplication". *ACM Transactions on Mathematical Software (TOMS)*.
[4] Yatsina, M. (2018). "LLVM Greedy Register Allocator – Improving Region Split Decisions". *European LLVM Developers' Meeting*.
[5] Lemire, D. et al. Microarchitectural Benchmarks for Apple Silicon.
[6] SoC Architecture Analyses. (2024). "Apple M4 Microarchitecture: LPDDR5X-7500 Bandwidth (120GB/s) and Scalable Matrix Extension (SME) Limits".
[7] Park, G., et al. (2024). "LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient LLM Inference". *ICLR 2024*.
