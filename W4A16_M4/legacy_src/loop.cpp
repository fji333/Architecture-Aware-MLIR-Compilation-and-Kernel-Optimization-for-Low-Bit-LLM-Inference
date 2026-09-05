#include <iostream>
#include<arm_neon.h>
int main(){
    int K = 4;
    float16_t act_matrix[4] = {
        1.0f, 2.0f, 3.0f, 4.0f
    };
    uint8_t weight_matrix[4*8];
    for(int i = 0; i < 32 ; i++){
        weight_matrix[i] = 0x5c;
    } ;
    float16x8_t v_scale = vdupq_n_f16(0.1f);
    float16x8_t v_bias =  vdupq_n_f16(-0.5f);
    float16x8_t v_accum_low = vdupq_n_f16(0.0f);
    float16x8_t v_accum_high = vdupq_n_f16(0.0f);
    float16_t* ptr_act = act_matrix;
    uint8_t* ptr_w = weight_matrix;
    for(int k = 0 ;k < K; k++){
        uint8x8_t v_w_packed = vld1_u8(ptr_w);
        ptr_w += 8;
        float16x8_t v_act = vld1q_dup_f16(ptr_act);
        ptr_act += 1;
        uint8x16_t v_w_128 = vcombine_u8(v_w_packed, vdup_n_u8(0));                                                                                                                                     
                                                                                                                                                                                                            
            // 提取低 4 位 (原本的 W0~W7)                                                                                                                                                                   
            uint8x16_t temp_low = vshlq_n_u8(v_w_128, 4);                                                                                                                                                   
            uint8x8_t v_low_int8 = vget_low_u8(vshrq_n_u8(temp_low, 4));                                                                                                                                    
                                                                                                                                                                                                            
            // 提取高 4 位 (原本的 W8~W15)                                                                                                                                                                  
            uint8x8_t v_high_int8 = vget_low_u8(vshrq_n_u8(v_w_128, 4));                                                                                                                                    
                                                                                                                                                                                                            
            // 转换与反量化乘加 (低 8 列)                                                                                                                                                                   
            float16x8_t v_w_fp16_low = vcvtq_f16_u16(vmovl_u8(v_low_int8));                                                                                                                                 
            v_w_fp16_low = vfmaq_f16(v_bias, v_w_fp16_low, v_scale);                                                                                                                                        
            v_accum_low  = vfmaq_f16(v_accum_low, v_w_fp16_low, v_act);                                                                                                                                     
                                                                                                                                                                                                            
            // 转换与反量化乘加 (高 8 列)                                                                                                                                                                   
            float16x8_t v_w_fp16_high = vcvtq_f16_u16(vmovl_u8(v_high_int8));                                                                                                                               
            v_w_fp16_high = vfmaq_f16(v_bias, v_w_fp16_high, v_scale);                                                                                                                                      
            v_accum_high  = vfmaq_f16(v_accum_high, v_w_fp16_high, v_act);  

    };
    std::cout << "k纬度循环跑完了"<< std::endl;
    return 0;
}