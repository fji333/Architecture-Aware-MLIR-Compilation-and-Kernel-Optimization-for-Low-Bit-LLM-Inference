#include <iostream>
#include <cstdint>
#include <iomanip>
int main(){
    int8_t w0 = 5;
    int8_t w1 = 12;
    uint8_t packed_byte = 0;
    packed_byte = ((w0 & 0x0F)<< 4 | (w1 & 0x0F));
    std::cout << "Packed byte (Hex) 0X "<< std::hex << (int)packed_byte << std::endl;
    return 0;

}

