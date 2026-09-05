#include <iostream>
#include <iomanip>
#include <arm_neon.h>
int main(){
    uint8_t packed_data[16] = {
        0xc5, 0xc5, 0xc5, 0xc5, 0xc5, 0xc5, 0xc5, 0xc5,                                                                                                                                              
        0xc5, 0xc5, 0xc5, 0xc5, 0xc5, 0xc5, 0xc5, 0xc5  };
    uint8x16_t v_unpacked = vld1q_u8(packed_data);
    uint8x16_t v_low_unpacked ;
    uint8x16_t v_high_unpacked;
    v_high_unpacked = vshrq_n_u8(v_unpacked,4);
    uint8x16_t temp = vshlq_n_u8(v_unpacked,4);
    v_low_unpacked = vshrq_n_u8(temp,4);
    uint8_t out_low[16];
    uint8_t out_high[16];
    vst1q_u8(out_low, v_low_unpacked);
    vst1q_u8(out_high, v_high_unpacked);
    std:: cout << "解包出的地位应该是 5 （0x05）:"<< (int)out_low[0]<<std::endl;
    std::cout << "解包出的高位应该是 12（0x0c）"<<(int)out_high[0] <<std::endl;
    return 0;
}