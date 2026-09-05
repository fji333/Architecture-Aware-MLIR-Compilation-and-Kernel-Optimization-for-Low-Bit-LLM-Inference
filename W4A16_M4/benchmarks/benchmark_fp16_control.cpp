#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <stdlib.h>
#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

// 纯 FP16 对照组：与 W4A16 微内核完全相同的结构（8x16 tile, Uk=2, 无分块, 线性扫描）
// 唯一区别：权重直接以 FP16 存储，内循环无任何反量化指令（vshlq/vshrq/vcvtq）
// 目的：通过消除反量化开销，验证 ALU 端口竞争是否为性能瓶颈的根因

void gemm_pure_fp16(const float16_t* A, const float16_t* W_fp16, float16_t* C, int M, int N, int K) {
    for (int m = 0; m < M; m += 8) {
        for (int n = 0; n < N; n += 16) {
            const float16_t* ptr_w = W_fp16 + n * K;  // 权重按列优先排列

            float16x8_t v_c0_low = vdupq_n_f16(0.0f); float16x8_t v_c0_high = vdupq_n_f16(0.0f);
            float16x8_t v_c1_low = vdupq_n_f16(0.0f); float16x8_t v_c1_high = vdupq_n_f16(0.0f);
            float16x8_t v_c2_low = vdupq_n_f16(0.0f); float16x8_t v_c2_high = vdupq_n_f16(0.0f);
            float16x8_t v_c3_low = vdupq_n_f16(0.0f); float16x8_t v_c3_high = vdupq_n_f16(0.0f);
            float16x8_t v_c4_low = vdupq_n_f16(0.0f); float16x8_t v_c4_high = vdupq_n_f16(0.0f);
            float16x8_t v_c5_low = vdupq_n_f16(0.0f); float16x8_t v_c5_high = vdupq_n_f16(0.0f);
            float16x8_t v_c6_low = vdupq_n_f16(0.0f); float16x8_t v_c6_high = vdupq_n_f16(0.0f);
            float16x8_t v_c7_low = vdupq_n_f16(0.0f); float16x8_t v_c7_high = vdupq_n_f16(0.0f);

            for (int k = 0; k < K; k += 2) {
                const float16_t* pA = A + m * K + k;

                // ======== K+0 ========
                // 直接加载 FP16 权重 —— 无反量化指令
                float16x8_t w_low_0  = vld1q_f16(ptr_w);        // 8 个 FP16
                float16x8_t w_high_0 = vld1q_f16(ptr_w + 8);    // 8 个 FP16

                float16x8_t a0_0 = vld1q_dup_f16(pA + 0 * K + 0);
                v_c0_low = vfmaq_f16(v_c0_low, w_low_0, a0_0); v_c0_high = vfmaq_f16(v_c0_high, w_high_0, a0_0);
                float16x8_t a1_0 = vld1q_dup_f16(pA + 1 * K + 0);
                v_c1_low = vfmaq_f16(v_c1_low, w_low_0, a1_0); v_c1_high = vfmaq_f16(v_c1_high, w_high_0, a1_0);
                float16x8_t a2_0 = vld1q_dup_f16(pA + 2 * K + 0);
                v_c2_low = vfmaq_f16(v_c2_low, w_low_0, a2_0); v_c2_high = vfmaq_f16(v_c2_high, w_high_0, a2_0);
                float16x8_t a3_0 = vld1q_dup_f16(pA + 3 * K + 0);
                v_c3_low = vfmaq_f16(v_c3_low, w_low_0, a3_0); v_c3_high = vfmaq_f16(v_c3_high, w_high_0, a3_0);
                float16x8_t a4_0 = vld1q_dup_f16(pA + 4 * K + 0);
                v_c4_low = vfmaq_f16(v_c4_low, w_low_0, a4_0); v_c4_high = vfmaq_f16(v_c4_high, w_high_0, a4_0);
                float16x8_t a5_0 = vld1q_dup_f16(pA + 5 * K + 0);
                v_c5_low = vfmaq_f16(v_c5_low, w_low_0, a5_0); v_c5_high = vfmaq_f16(v_c5_high, w_high_0, a5_0);
                float16x8_t a6_0 = vld1q_dup_f16(pA + 6 * K + 0);
                v_c6_low = vfmaq_f16(v_c6_low, w_low_0, a6_0); v_c6_high = vfmaq_f16(v_c6_high, w_high_0, a6_0);
                float16x8_t a7_0 = vld1q_dup_f16(pA + 7 * K + 0);
                v_c7_low = vfmaq_f16(v_c7_low, w_low_0, a7_0); v_c7_high = vfmaq_f16(v_c7_high, w_high_0, a7_0);

                // ======== K+1 ========
                float16x8_t w_low_1  = vld1q_f16(ptr_w + 16);
                float16x8_t w_high_1 = vld1q_f16(ptr_w + 24);

                float16x8_t a0_1 = vld1q_dup_f16(pA + 0 * K + 1);
                v_c0_low = vfmaq_f16(v_c0_low, w_low_1, a0_1); v_c0_high = vfmaq_f16(v_c0_high, w_high_1, a0_1);
                float16x8_t a1_1 = vld1q_dup_f16(pA + 1 * K + 1);
                v_c1_low = vfmaq_f16(v_c1_low, w_low_1, a1_1); v_c1_high = vfmaq_f16(v_c1_high, w_high_1, a1_1);
                float16x8_t a2_1 = vld1q_dup_f16(pA + 2 * K + 1);
                v_c2_low = vfmaq_f16(v_c2_low, w_low_1, a2_1); v_c2_high = vfmaq_f16(v_c2_high, w_high_1, a2_1);
                float16x8_t a3_1 = vld1q_dup_f16(pA + 3 * K + 1);
                v_c3_low = vfmaq_f16(v_c3_low, w_low_1, a3_1); v_c3_high = vfmaq_f16(v_c3_high, w_high_1, a3_1);
                float16x8_t a4_1 = vld1q_dup_f16(pA + 4 * K + 1);
                v_c4_low = vfmaq_f16(v_c4_low, w_low_1, a4_1); v_c4_high = vfmaq_f16(v_c4_high, w_high_1, a4_1);
                float16x8_t a5_1 = vld1q_dup_f16(pA + 5 * K + 1);
                v_c5_low = vfmaq_f16(v_c5_low, w_low_1, a5_1); v_c5_high = vfmaq_f16(v_c5_high, w_high_1, a5_1);
                float16x8_t a6_1 = vld1q_dup_f16(pA + 6 * K + 1);
                v_c6_low = vfmaq_f16(v_c6_low, w_low_1, a6_1); v_c6_high = vfmaq_f16(v_c6_high, w_high_1, a6_1);
                float16x8_t a7_1 = vld1q_dup_f16(pA + 7 * K + 1);
                v_c7_low = vfmaq_f16(v_c7_low, w_low_1, a7_1); v_c7_high = vfmaq_f16(v_c7_high, w_high_1, a7_1);

                ptr_w += 32;  // 2 steps * 16 FP16 values
            }

            float16_t* ptr_out = C + (m * N) + n;
            vst1q_f16(ptr_out + 0 * N, v_c0_low); vst1q_f16(ptr_out + 0 * N + 8, v_c0_high);
            vst1q_f16(ptr_out + 1 * N, v_c1_low); vst1q_f16(ptr_out + 1 * N + 8, v_c1_high);
            vst1q_f16(ptr_out + 2 * N, v_c2_low); vst1q_f16(ptr_out + 2 * N + 8, v_c2_high);
            vst1q_f16(ptr_out + 3 * N, v_c3_low); vst1q_f16(ptr_out + 3 * N + 8, v_c3_high);
            vst1q_f16(ptr_out + 4 * N, v_c4_low); vst1q_f16(ptr_out + 4 * N + 8, v_c4_high);
            vst1q_f16(ptr_out + 5 * N, v_c5_low); vst1q_f16(ptr_out + 5 * N + 8, v_c5_high);
            vst1q_f16(ptr_out + 6 * N, v_c6_low); vst1q_f16(ptr_out + 6 * N + 8, v_c6_high);
            vst1q_f16(ptr_out + 7 * N, v_c7_low); vst1q_f16(ptr_out + 7 * N + 8, v_c7_high);
        }
    }
}

int main() {
    int M = 1024;
    int N = 16384;
    int K = 4096;

#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    float16_t* A = nullptr;
    float16_t* W_fp16 = nullptr;  // 权重直接用 FP16，不做 4-bit 压缩
    float16_t* C = nullptr;
    posix_memalign((void**)&A, 16384, M * K * sizeof(float16_t));
    posix_memalign((void**)&W_fp16, 16384, N * K * sizeof(float16_t));  // 注意: 体积是 W4 的 4 倍
    posix_memalign((void**)&C, 16384, M * N * sizeof(float16_t));

    std::fill(A, A + M * K, 1.0f);
    std::fill(W_fp16, W_fp16 + N * K, 0.5f);
    std::fill(C, C + M * N, 0.0f);

    // Warmup
    gemm_pure_fp16(A, W_fp16, C, M, N, K);

    int num_runs = 3;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        gemm_pure_fp16(A, W_fp16, C, M, N, K);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double avg_time_sec = diff.count() / num_runs;

    double total_flops = 2.0 * M * N * K;
    double gflops = (total_flops / avg_time_sec) / 1e9;

    std::cout << "===== 纯 FP16 对照组 (无反量化指令) =====" << std::endl;
    std::cout << "耗时: " << avg_time_sec * 1000.0 << " ms, ";
    std::cout << "算力: " << gflops << " GFLOPS" << std::endl;
    std::cout << "W4A16 基准: ~155 GFLOPS (含 vshlq/vshrq/vcvtq 反量化)" << std::endl;
    std::cout << "若纯 FP16 显著高于 155，则证明 ALU 端口竞争是瓶颈根因。" << std::endl;

    free(A);
    free(W_fp16);
    free(C);

    return 0;
}
