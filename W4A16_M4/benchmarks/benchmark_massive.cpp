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

#ifndef L2_MC
#define L2_MC 0
#endif
#ifndef L2_NC
#define L2_NC 0
#endif
#ifndef L1_MC
#define L1_MC 0
#endif
#ifndef L1_NC
#define L1_NC 0
#endif
#ifndef UK
#define UK 2
#endif

void gemm_m4_w4a16_ablation(const float16_t* A, const uint8_t* W, float16_t* C, int M, int N, int K) {
    float16x8_t v_scale = vdupq_n_f16(0.1f);  
    float16x8_t v_bias  = vdupq_n_f16(-0.5f); 

    int l2_mc_step = (L2_MC > 0) ? L2_MC : M;
    int l2_nc_step = (L2_NC > 0) ? L2_NC : N;
    
    // 如果没有指定 L1 分块，默认 L1 的步长等于 L2，相当于跳过 L1 这一层
    int l1_mc_step = (L1_MC > 0) ? L1_MC : l2_mc_step;
    int l1_nc_step = (L1_NC > 0) ? L1_NC : l2_nc_step;

    for (int l2_mc = 0; l2_mc < M; l2_mc += l2_mc_step) {
        for (int l2_nc = 0; l2_nc < N; l2_nc += l2_nc_step) {
            
            for (int l1_mc = l2_mc; l1_mc < std::min(l2_mc + l2_mc_step, M); l1_mc += l1_mc_step) {
                for (int l1_nc = l2_nc; l1_nc < std::min(l2_nc + l2_nc_step, N); l1_nc += l1_nc_step) {
                    
                    for (int m = l1_mc; m < std::min(l1_mc + l1_mc_step, M); m += 8) {
                        for (int n = l1_nc; n < std::min(l1_nc + l1_nc_step, N); n += 16) {
                            
                            const uint8_t* ptr_w = W + (n / 16) * (K * 8); 

                            float16x8_t v_c0_low = vdupq_n_f16(0.0f); float16x8_t v_c0_high = vdupq_n_f16(0.0f);
                            float16x8_t v_c1_low = vdupq_n_f16(0.0f); float16x8_t v_c1_high = vdupq_n_f16(0.0f);
                            float16x8_t v_c2_low = vdupq_n_f16(0.0f); float16x8_t v_c2_high = vdupq_n_f16(0.0f);
                            float16x8_t v_c3_low = vdupq_n_f16(0.0f); float16x8_t v_c3_high = vdupq_n_f16(0.0f);
                            float16x8_t v_c4_low = vdupq_n_f16(0.0f); float16x8_t v_c4_high = vdupq_n_f16(0.0f);
                            float16x8_t v_c5_low = vdupq_n_f16(0.0f); float16x8_t v_c5_high = vdupq_n_f16(0.0f);
                            float16x8_t v_c6_low = vdupq_n_f16(0.0f); float16x8_t v_c6_high = vdupq_n_f16(0.0f);
                            float16x8_t v_c7_low = vdupq_n_f16(0.0f); float16x8_t v_c7_high = vdupq_n_f16(0.0f);

                            for (int k = 0; k < K; k += UK) {
                                const float16_t* pA = A + m * K + k;
                                const uint8_t* pW = ptr_w;

                                #define PROCESS_K(step) \
                                    uint8x8_t w##step = vld1_u8(pW + step * 8); \
                                    uint8x16_t w128_##step = vcombine_u8(w##step, vdup_n_u8(0)); \
                                    float16x8_t w_low_##step = vfmaq_f16(v_bias, vcvtq_f16_u16(vmovl_u8(vget_low_u8(vshrq_n_u8(vshlq_n_u8(w128_##step, 4), 4)))), v_scale); \
                                    float16x8_t w_high_##step = vfmaq_f16(v_bias, vcvtq_f16_u16(vmovl_u8(vget_low_u8(vshrq_n_u8(w128_##step, 4)))), v_scale); \
                                    \
                                    float16x8_t a0_##step = vld1q_dup_f16(pA + 0 * K + step); \
                                    v_c0_low = vfmaq_f16(v_c0_low, w_low_##step, a0_##step); v_c0_high = vfmaq_f16(v_c0_high, w_high_##step, a0_##step); \
                                    float16x8_t a1_##step = vld1q_dup_f16(pA + 1 * K + step); \
                                    v_c1_low = vfmaq_f16(v_c1_low, w_low_##step, a1_##step); v_c1_high = vfmaq_f16(v_c1_high, w_high_##step, a1_##step); \
                                    float16x8_t a2_##step = vld1q_dup_f16(pA + 2 * K + step); \
                                    v_c2_low = vfmaq_f16(v_c2_low, w_low_##step, a2_##step); v_c2_high = vfmaq_f16(v_c2_high, w_high_##step, a2_##step); \
                                    float16x8_t a3_##step = vld1q_dup_f16(pA + 3 * K + step); \
                                    v_c3_low = vfmaq_f16(v_c3_low, w_low_##step, a3_##step); v_c3_high = vfmaq_f16(v_c3_high, w_high_##step, a3_##step); \
                                    float16x8_t a4_##step = vld1q_dup_f16(pA + 4 * K + step); \
                                    v_c4_low = vfmaq_f16(v_c4_low, w_low_##step, a4_##step); v_c4_high = vfmaq_f16(v_c4_high, w_high_##step, a4_##step); \
                                    float16x8_t a5_##step = vld1q_dup_f16(pA + 5 * K + step); \
                                    v_c5_low = vfmaq_f16(v_c5_low, w_low_##step, a5_##step); v_c5_high = vfmaq_f16(v_c5_high, w_high_##step, a5_##step); \
                                    float16x8_t a6_##step = vld1q_dup_f16(pA + 6 * K + step); \
                                    v_c6_low = vfmaq_f16(v_c6_low, w_low_##step, a6_##step); v_c6_high = vfmaq_f16(v_c6_high, w_high_##step, a6_##step); \
                                    float16x8_t a7_##step = vld1q_dup_f16(pA + 7 * K + step); \
                                    v_c7_low = vfmaq_f16(v_c7_low, w_low_##step, a7_##step); v_c7_high = vfmaq_f16(v_c7_high, w_high_##step, a7_##step);

                                #if UK >= 1
                                PROCESS_K(0)
                                #endif
                                #if UK >= 2
                                PROCESS_K(1)
                                #endif
                                #if UK >= 3
                                PROCESS_K(2)
                                #endif
                                #if UK >= 4
                                PROCESS_K(3)
                                #endif
                                #if UK >= 5
                                PROCESS_K(4)
                                #endif
                                #if UK >= 6
                                PROCESS_K(5)
                                #endif
                                #if UK >= 7
                                PROCESS_K(6)
                                #endif
                                #if UK >= 8
                                PROCESS_K(7)
                                #endif
                                
                                ptr_w += UK * 8;
                            }

                            // 3. 将物理寄存器写回主内存
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
            }
        }
    }
}

int main() {
    int M = 1024; 
    int N = 16384; 
    int K = 4096;

#if defined(__APPLE__)
    // 强制绑定 P-Core 最高优先级，防止微基准测试由于降频或调度到 E-Core 导致的随机波动
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    // 使用 posix_memalign 强制执行 16KB 操作系统页对齐，彻底消除 vld1q 的跨缓存行加载惩罚
    float16_t* A = nullptr;
    uint8_t* W = nullptr;
    float16_t* C = nullptr;
    posix_memalign((void**)&A, 16384, M * K * sizeof(float16_t));
    posix_memalign((void**)&W, 16384, K * N / 2 * sizeof(uint8_t));
    posix_memalign((void**)&C, 16384, M * N * sizeof(float16_t));

    std::fill(A, A + M * K, 1.0f);
    std::fill(W, W + K * N / 2, 0x5c);
    std::fill(C, C + M * N, 0.0f);

    gemm_m4_w4a16_ablation(A, W, C, M, N, K);

    int num_runs = 3;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        gemm_m4_w4a16_ablation(A, W, C, M, N, K);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end_time - start_time;
    double avg_time_sec = diff.count() / num_runs;
    
    double total_flops = 2.0 * M * N * K; 
    double gflops = (total_flops / avg_time_sec) / 1e9;

    std::cout << "耗时: " << avg_time_sec * 1000.0 << " ms, ";
    std::cout << "算力: " << gflops << " GFLOPS\n";

    free(A);
    free(W);
    free(C);

    return 0;
}
