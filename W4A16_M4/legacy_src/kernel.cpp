#include <iostream>
#include <iomanip>
#include <arm_neon.h>

int main(){
    uint8_t low_data[16] = {5, 5, 5, 5, 5, 5, 5, 5, 5};

    uint8x8_t v_low_int8 = vld1_u8(low_data);

    float16_t act = 2.0f;

    float16x8_t v_act = vdupq_n_f16(act);

    float16x8_t v_scale = vdupq_n_f16(0.1f);
    
    float16x8_t v_bias = vdupq_n_f16(-0.5f);
    float16x8_t v_accmulate = vdupq_n_f16(0.0f);
    uint16x8_t v_low_int16 = vmovl_u8(v_low_int8);
    float16x8_t v_weight_fp16 = vcvtq_f16_u16(v_low_int16);
    v_weight_fp16 = vfmaq_f16(v_bias,v_weight_fp16,v_scale);
    v_accmulate = vfmaq_f16(v_accmulate, v_weight_fp16, v_act);
    float16_t out_res[8];
    vst1q_f16(out_res, v_weight_fp16);
    std::cout << "FMA内核的计算结果是" << std::endl;
    std::cout << "最终累加器的第一个值应为0:"<< (float)out_res[0]<< std::endl;
    return 0;
}