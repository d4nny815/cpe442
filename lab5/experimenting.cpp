#include <stdio.h>  
#include <cstdint>
#include <arm_neon.h>
#include <opencv2/opencv.hpp>


int main(int argc, char** argv) {

    // initialize
    const int SIZE = 32;
    uint8_t arr1[SIZE];
    uint8_t arr2[SIZE];

    for (int i = 0; i < SIZE; i++) {
        arr1[i] = i;
        arr2[i] = 2 * i;
    }

    // scalar mode
    uint8_t arr3[SIZE];
    for (int i = 0; i < SIZE; i++) {
        arr3[i] = arr1[i] + arr2[i];
    }

    // vector mode
    uint8_t arrv[SIZE];
    for (int i = 0; i < SIZE / 8; i++) {
        uint8x8_t arrv1 = vld1_u8(arr1 + i * 8);
        uint8x8_t arrv2 = vld1_u8(arr2 + i * 8);
        uint8x8_t arrv3 = vadd_u8(arrv1, arrv2);
        vst1_u8(arrv + i * 8, arrv3);
    }

    // display
    for (int i = 0; i < SIZE; i++) {
        printf("%u + %u = %u = %u\n", arr1[i], arr2[i], arr3[i], arrv[i]);
    }

    return 0;
}