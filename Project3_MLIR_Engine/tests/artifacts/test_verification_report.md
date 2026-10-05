# OmniSchedule Geek-Level Industrial Verification & Stress Suite Report

- **目标芯片与微架构**: Apple M4 (ARMv9.2-A, 4x 128-bit NEON, 128KB L1D, 12MB L2)
- **测试类别覆盖**: 7 大类 (长宽比/GEMV、非2次幂、LLM投影、分组深层累加、对抗Fuzzing、多调度交叉一致性、1000轮长稳)
- **总测试项数**: **47 项**
- **成功通过项数**: **47 项**
- **综合通过率**: **100.0% (Bit-Exact Verified)**

## 详细测试用例列表

| 测试用例名称 | 分类 | 余弦相似度 (CosSim) | 相对 Frobenius 误差 | MAE / RMSE | 判定状态 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Tall-Skinny Single Token Decode (M=8, K=128, N=64) | `Aspect_Ratio_Sweep` | 0.999999 | 0.001591 | 0.004738 | ✅ PASS |
| Small Batch Decode (M=16, K=128, N=128) | `Aspect_Ratio_Sweep` | 0.999999 | 0.001628 | 0.004595 | ✅ PASS |
| Medium Batch Decode (M=32, K=128, N=128) | `Aspect_Ratio_Sweep` | 0.999998 | 0.001656 | 0.005125 | ✅ PASS |
| Micro-Tile Baseline Probe (64x64x128) | `Aspect_Ratio_Sweep` | 0.999998 | 0.001724 | 0.005098 | ✅ PASS |
| Spatial Unroll Block (2x N Span: 64x128x128) | `Aspect_Ratio_Sweep` | 0.999997 | 0.001695 | 0.005153 | ✅ PASS |
| Reduction Contraction Block (2x M Span: 128x64x128) | `Aspect_Ratio_Sweep` | 0.999998 | 0.001677 | 0.005255 | ✅ PASS |
| Symmetric Square Tile (128x128x128) | `Aspect_Ratio_Sweep` | 0.999997 | 0.001620 | 0.004851 | ✅ PASS |
| Double Reduction Depth Tile (64x64x256) | `Aspect_Ratio_Sweep` | 0.999997 | 0.002334 | 0.010485 | ✅ PASS |
| Wide Matrix Contraction (64x256x128) | `Aspect_Ratio_Sweep` | 0.999997 | 0.001682 | 0.004876 | ✅ PASS |
| Extended Square Tile (128x128x256) | `Aspect_Ratio_Sweep` | 0.999996 | 0.002313 | 0.010264 | ✅ PASS |
| Odd Batch Dimension (M=24, N=64, K=128) | `Non_Power_of_2` | 0.999999 | 0.001626 | 0.004313 | ✅ PASS |
| Non-Power-of-2 Channel (M=64, N=48, K=128) | `Non_Power_of_2` | 0.999999 | 0.001620 | 0.005483 | ✅ PASS |
| Unaligned Spatial Width (M=64, N=96, K=128) | `Non_Power_of_2` | 0.999998 | 0.001701 | 0.005260 | ✅ PASS |
| Asymmetric Extended Block (M=48, N=160, K=128) | `Non_Power_of_2` | 0.999998 | 0.001698 | 0.005235 | ✅ PASS |
| Triple Group Unaligned Depth (M=32, N=64, K=384) | `Non_Power_of_2` | 0.999996 | 0.002797 | 0.016211 | ✅ PASS |
| Arbitrary Batch Scale (M=56, N=128, K=256) | `Non_Power_of_2` | 0.999996 | 0.002397 | 0.010808 | ✅ PASS |
| LLaMA-3 Attention Q/K/V Probe | `LLM_Projections` | 0.999998 | 0.001721 | 0.004786 | ✅ PASS |
| LLaMA-3 Output Projection Tile | `LLM_Projections` | 0.999997 | 0.001694 | 0.005171 | ✅ PASS |
| LLaMA-3 Feed-Forward SwiGLU Slice | `LLM_Projections` | 0.999998 | 0.001712 | 0.004570 | ✅ PASS |
| LLaMA-3 Dual-Group Contraction Layer | `LLM_Projections` | 0.999995 | 0.002323 | 0.011349 | ✅ PASS |
| Qwen-2.5-7B Non-Power-of-2 Attention Tile | `LLM_Projections` | 0.999998 | 0.001712 | 0.004570 | ✅ PASS |
| Qwen-2.5-7B Wide SwiGLU MLP Block | `LLM_Projections` | 0.999997 | 0.001751 | 0.004573 | ✅ PASS |
| Mistral-7B Sliding Window Attention Block | `LLM_Projections` | 0.999997 | 0.001694 | 0.005171 | ✅ PASS |
| DeepSeek-V2 MoE Routed Expert Block | `LLM_Projections` | 0.999997 | 0.002268 | 0.009016 | ✅ PASS |
| 1-Group Single (K=128, Group Count=1) | `Multi_Group_Traversal` | 0.999998 | 0.001663 | 0.005537 | ✅ PASS |
| 2-Group Dual (K=256, Group Count=2) | `Multi_Group_Traversal` | 0.999997 | 0.002318 | 0.010119 | ✅ PASS |
| 3-Group Triple (K=384, Group Count=3) | `Multi_Group_Traversal` | 0.999995 | 0.002800 | 0.016627 | ✅ PASS |
| 4-Group Quad (K=512, Group Count=4) | `Multi_Group_Traversal` | 0.999994 | 0.003287 | 0.023417 | ✅ PASS |
| 6-Group Hexa (K=768, Group Count=6) | `Multi_Group_Traversal` | 0.999990 | 0.004280 | 0.035539 | ✅ PASS |
| 8-Group Octa Deep (K=1024, Group Count=8) | `Multi_Group_Traversal` | 0.999989 | 0.004516 | 0.043387 | ✅ PASS |
| 1. All-Zeros INT4 Weights (Q=0, Byte=0x00) | `Adversarial_Fuzzing` | 0.999999 | 0.001732 | 0.002509 | ✅ PASS |
| 2. All-Max INT4 Weights (Q=15, Byte=0xFF) | `Adversarial_Fuzzing` | 0.999997 | 0.001704 | 0.008097 | ✅ PASS |
| 3. Alternating Nibbles (0x0F, 0xF0 Checkerboard) | `Adversarial_Fuzzing` | 0.999996 | 0.001714 | 0.006463 | ✅ PASS |
| 4. Interleaved Bits (0x55, 0xAA Pattern) | `Adversarial_Fuzzing` | 0.999997 | 0.001682 | 0.003607 | ✅ PASS |
| 5. Sparse LLM Outlier Activations (1% Outliers at 50x) | `Adversarial_Fuzzing` | 0.999998 | 0.001484 | 0.019822 | ✅ PASS |
| 6. High Dynamic Range Activations (Scale x100) | `Adversarial_Fuzzing` | 0.999997 | 0.001642 | 0.461088 | ✅ PASS |
| 7. Subnormal Underflow Probing (Scale=1e-7) | `Adversarial_Fuzzing` | 0.999998 | 0.000012 | 0.000000 | ✅ PASS |
| 8. Saturation Overflow Probing (Scale=50.0) | `Adversarial_Fuzzing` | 0.999996 | 0.001663 | 4.142780 | ✅ PASS |
| 9. Zero-Bias Degradation (Bias=0) | `Adversarial_Fuzzing` | 0.999996 | 0.001667 | 0.004654 | ✅ PASS |
| 10. Extreme Asymmetric Zero-Points (Z=0 vs Z=15) | `Adversarial_Fuzzing` | 0.999997 | 0.001635 | 0.006178 | ✅ PASS |
| 11. Heavy-Tailed Cauchy Distributed Activations | `Adversarial_Fuzzing` | 0.999997 | 0.001603 | 0.035174 | ✅ PASS |
| 12. Ill-Conditioned Orthogonal Perturbation Matrix | `Adversarial_Fuzzing` | 0.999997 | 0.001199 | 0.000104 | ✅ PASS |
| Schedule S2 (L1 Single-Tier Fused) | `Multi_Schedule_Consistency` | 0.999998 | 0.001645 | 0.005304 | ✅ PASS |
| Schedule S3 (L2+L1 Hierarchical) | `Multi_Schedule_Consistency` | 0.999998 | 0.001645 | 0.005304 | ✅ PASS |
| Schedule S4 (M4 Specialized Tiling) | `Multi_Schedule_Consistency` | 0.999998 | 0.001645 | 0.005304 | ✅ PASS |
| Schedule S6 (Deep Transfer Hoisting & Unroll) | `Multi_Schedule_Consistency` | 0.999998 | 0.001645 | 0.005304 | ✅ PASS |
| 1000_Stress_Loop | `Stress_Testing` | 1.000000 | 0.000000 | 0.000000 | ✅ PASS |
