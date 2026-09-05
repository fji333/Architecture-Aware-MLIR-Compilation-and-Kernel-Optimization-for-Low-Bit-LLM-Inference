#include<iostream>
#include<arm_neon.h>
int main(){
    int K = 4;
    float16_t act_matrix[8*4];
    uint8_t weight_matrix[8*4];
    for(int i = 0 ;i < 32;i++)
    {
        weight_matrix[i] = 0x5c;
        act_matrix[i] = 0.1f;
    };
    float16x8_t v_c0_low = vdupq_n_f16(0.0f);
    float16x8_t v_c0_high = vdupq_n_f16(0.0f);
    float16x8_t v_c1_low = vdupq_n_f16(0.0f);
    float16x8_t v_c1_high = vdupq_n_f16(0.0f);
    float16x8_t v_c2_low = vdupq_n_f16(0.0f);
    float16x8_t v_c2_high = vdupq_n_f16(0.0f);
    float16x8_t v_c3_low = vdupq_n_f16(0.0f);
    float16x8_t v_c3_high = vdupq_n_f16(0.0f);
    float16x8_t v_c4_low = vdupq_n_f16(0.0f);
    float16x8_t v_c4_high = vdupq_n_f16(0.0f);
    float16x8_t v_c5_low = vdupq_n_f16(0.0f);
    float16x8_t v_c5_high = vdupq_n_f16(0.0f);
    float16x8_t v_c6_low = vdupq_n_f16(0.0f);
    float16x8_t v_c6_high = vdupq_n_f16(0.0f);
    float16x8_t v_c7_low = vdupq_n_f16(0.0f);
    float16x8_t v_c7_high = vdupq_n_f16(0.0f);
    float16x8_t v_scale =vdupq_n_f16(0.1f);
    float16x8_t v_bias = vdupq_n_f16(-0.5f);
    float16_t* ptr_act = act_matrix;
    uint8_t* ptr_w = weight_matrix;
    for(int k = 0; k<K; k += 2){
            uint8x8_t v_w_packed_0 = vld1_u8(ptr_w);                                                                                                                                                        
            uint8x8_t v_w_packed_1 = vld1_u8(ptr_w + 8);                                                                                                                                                    
            ptr_w += 16;                                                                                                                                                                                    
                                                                                                                                                                                                            
            // 并发拆分 128 bit                                                                                                                                                                             
            uint8x16_t v_w_128_0 = vcombine_u8(v_w_packed_0, vdup_n_u8(0));                                                                                                                                 
            uint8x16_t v_w_128_1 = vcombine_u8(v_w_packed_1, vdup_n_u8(0));                                                                                                                                 
                                                                                                                                                                                                            
            // 并发移位提取高低 4-bit                                                                                                                                                                       
            uint8x8_t v_low_int8_0  = vget_low_u8(vshrq_n_u8(vshlq_n_u8(v_w_128_0, 4), 4));                                                                                                                 
            uint8x8_t v_high_int8_0 = vget_low_u8(vshrq_n_u8(v_w_128_0, 4));                                                                                                                                
            uint8x8_t v_low_int8_1  = vget_low_u8(vshrq_n_u8(vshlq_n_u8(v_w_128_1, 4), 4));                                                                                                                 
            uint8x8_t v_high_int8_1 = vget_low_u8(vshrq_n_u8(v_w_128_1, 4));                                                                                                                                
                                                                                                                                                                                                            
            // 并发转换与 Scale/Bias 运算                                                                                                                                                                   
            float16x8_t v_w_fp16_low_0  = vfmaq_f16(v_bias, vcvtq_f16_u16(vmovl_u8(v_low_int8_0)),  v_scale);                                                                                               
            float16x8_t v_w_fp16_high_0 = vfmaq_f16(v_bias, vcvtq_f16_u16(vmovl_u8(v_high_int8_0)), v_scale);                                                                                               
            float16x8_t v_w_fp16_low_1  = vfmaq_f16(v_bias, vcvtq_f16_u16(vmovl_u8(v_low_int8_1)),  v_scale);                                                                                               
            float16x8_t v_w_fp16_high_1 = vfmaq_f16(v_bias, vcvtq_f16_u16(vmovl_u8(v_high_int8_1)), v_scale);                                                                                               
                                                                                                                                                                                                            
            // --- 阶段 B：火力倾泻（让编译器织拉链的地方） ---                                                                                                                                             
                                                                                                                                                                                                            
            // Row 0                                                                                                                                                                                        
            float16x8_t v_act0_0 = vld1q_dup_f16(ptr_act + 0);                                                                                                                                              
            float16x8_t v_act0_1 = vld1q_dup_f16(ptr_act + 8);                                                                                                                                              
            v_c0_low  = vfmaq_f16(v_c0_low, v_w_fp16_low_0,  v_act0_0);                                                                                                                                     
            v_c0_low  = vfmaq_f16(v_c0_low, v_w_fp16_low_1,  v_act0_1);                                                                                                                                     
            v_c0_high = vfmaq_f16(v_c0_high, v_w_fp16_high_0, v_act0_0);                                                                                                                                    
            v_c0_high = vfmaq_f16(v_c0_high, v_w_fp16_high_1, v_act0_1);                                                                                                                                    
                                                                                                                                                                                                            
            // Row 1                                                                                                                                                                                        
            float16x8_t v_act1_0 = vld1q_dup_f16(ptr_act + 1);                                                                                                                                              
            float16x8_t v_act1_1 = vld1q_dup_f16(ptr_act + 9);                                                                                                                                              
            v_c1_low  = vfmaq_f16(v_c1_low, v_w_fp16_low_0,  v_act1_0);                                                                                                                                     
            v_c1_low  = vfmaq_f16(v_c1_low, v_w_fp16_low_1,  v_act1_1);                                                                                                                                     
            v_c1_high = vfmaq_f16(v_c1_high, v_w_fp16_high_0, v_act1_0);                                                                                                                                    
            v_c1_high = vfmaq_f16(v_c1_high, v_w_fp16_high_1, v_act1_1);                                                                                                                                    
                                                                                                                                                                                                            
            // Row 2                                                                                                                                                                                        
            float16x8_t v_act2_0 = vld1q_dup_f16(ptr_act + 2);                                                                                                                                              
            float16x8_t v_act2_1 = vld1q_dup_f16(ptr_act + 10);                                                                                                                                             
            v_c2_low  = vfmaq_f16(v_c2_low, v_w_fp16_low_0,  v_act2_0);                                                                                                                                     
            v_c2_low  = vfmaq_f16(v_c2_low, v_w_fp16_low_1,  v_act2_1);                                                                                                                                     
            v_c2_high = vfmaq_f16(v_c2_high, v_w_fp16_high_0, v_act2_0);                                                                                                                                    
            v_c2_high = vfmaq_f16(v_c2_high, v_w_fp16_high_1, v_act2_1);                                                                                                                                    
                                                                                                                                                                                                            
            // Row 3                                                                                                                                                                                        
            float16x8_t v_act3_0 = vld1q_dup_f16(ptr_act + 3);                                                                                                                                              
            float16x8_t v_act3_1 = vld1q_dup_f16(ptr_act + 11);                                                                                                                                             
            v_c3_low  = vfmaq_f16(v_c3_low, v_w_fp16_low_0,  v_act3_0);                                                                                                                                     
            v_c3_low  = vfmaq_f16(v_c3_low, v_w_fp16_low_1,  v_act3_1);                                                                                                                                     
            v_c3_high = vfmaq_f16(v_c3_high, v_w_fp16_high_0, v_act3_0);                                                                                                                                    
            v_c3_high = vfmaq_f16(v_c3_high, v_w_fp16_high_1, v_act3_1);                                                                                                                                    
                                                                                                                                                                                                            
            // Row 4                                                                                                                                                                                        
            float16x8_t v_act4_0 = vld1q_dup_f16(ptr_act + 4);                                                                                                                                              
            float16x8_t v_act4_1 = vld1q_dup_f16(ptr_act + 12);                                                                                                                                             
            v_c4_low  = vfmaq_f16(v_c4_low, v_w_fp16_low_0,  v_act4_0);                                                                                                                                     
            v_c4_low  = vfmaq_f16(v_c4_low, v_w_fp16_low_1,  v_act4_1);                                                                                                                                     
            v_c4_high = vfmaq_f16(v_c4_high, v_w_fp16_high_0, v_act4_0);                                                                                                                                    
            v_c4_high = vfmaq_f16(v_c4_high, v_w_fp16_high_1, v_act4_1);                                                                                                                                    
                                                                                                                                                                                                            
            // Row 5                                                                                                                                                                                        
            float16x8_t v_act5_0 = vld1q_dup_f16(ptr_act + 5);                                                                                                                                              
            float16x8_t v_act5_1 = vld1q_dup_f16(ptr_act + 13);                                                                                                                                             
            v_c5_low  = vfmaq_f16(v_c5_low, v_w_fp16_low_0,  v_act5_0);                                                                                                                                     
            v_c5_low  = vfmaq_f16(v_c5_low, v_w_fp16_low_1,  v_act5_1);                                                                                                                                     
            v_c5_high = vfmaq_f16(v_c5_high, v_w_fp16_high_0, v_act5_0);                                                                                                                                    
            v_c5_high = vfmaq_f16(v_c5_high, v_w_fp16_high_1, v_act5_1);                                                                                                                                    
                                                                                                                                                                                                            
            // Row 6                                                                                                                                                                                        
            float16x8_t v_act6_0 = vld1q_dup_f16(ptr_act + 6);                                                                                                                                              
            float16x8_t v_act6_1 = vld1q_dup_f16(ptr_act + 14);                                                                                                                                             
            v_c6_low  = vfmaq_f16(v_c6_low, v_w_fp16_low_0,  v_act6_0);                                                                                                                                     
            v_c6_low  = vfmaq_f16(v_c6_low, v_w_fp16_low_1,  v_act6_1);                                                                                                                                     
            v_c6_high = vfmaq_f16(v_c6_high, v_w_fp16_high_0, v_act6_0);                                                                                                                                    
            v_c6_high = vfmaq_f16(v_c6_high, v_w_fp16_high_1, v_act6_1);                                                                                                                                    
                                                                                                                                                                                                            
            // Row 7                                                                                                                                                                                        
            float16x8_t v_act7_0 = vld1q_dup_f16(ptr_act + 7);                                                                                                                                              
            float16x8_t v_act7_1 = vld1q_dup_f16(ptr_act + 15);                                                                                                                                             
            v_c7_low  = vfmaq_f16(v_c7_low, v_w_fp16_low_0,  v_act7_0);                                                                                                                                     
            v_c7_low  = vfmaq_f16(v_c7_low, v_w_fp16_low_1,  v_act7_1);                                                                                                                                     
            v_c7_high = vfmaq_f16(v_c7_high, v_w_fp16_high_0, v_act7_0);                                                                                                                                    
            v_c7_high = vfmaq_f16(v_c7_high, v_w_fp16_high_1, v_act7_1);                                                                                                                                    
                                                                                                                                                                                                            
            // 步进 16 个身位                                                                                                                          
            ptr_act += 16;                            

    };
    std::cout<<"框架搭建完毕"<<std::endl;
    return 0;

}

