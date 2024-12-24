#include "sobel.hpp"

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdint.h>

using namespace cv;
using namespace std;

inline uint8_t apply_greyscale(uint8_t* pixel) {
    return RED_WEIGHT * pixel[2] + GREEN_WEIGHT * pixel[1] + BLUE_WEIGHT * pixel[0];
}


void to442_greyscale(Mat& frame, Mat& end_frame) {
    for (int row = 0; row < frame.rows; row++) {
        uint8_t* frame_row = frame.ptr<uint8_t>(row);
        uint8_t* grey_row = end_frame.ptr<uint8_t>(row);
        
        for (int col = 0; col < frame.cols; col++) {
            grey_row[col] = apply_greyscale(&frame_row[col * BYTES_PER_PIXEL]);
        }
    }

    return; 
}


void get_neighbors(Mat& frame, int row, int col, uint8_t* neighbors_arr) {
    uint8_t* frame_data = frame.ptr<uint8_t>();
    int cols = frame.cols;
    int idx = row * cols + col;

    neighbors_arr[0] = frame_data[idx - cols - 1]; // top-left
    neighbors_arr[1] = frame_data[idx - cols];     // top-center
    neighbors_arr[2] = frame_data[idx - cols + 1]; // top-right
    neighbors_arr[3] = frame_data[idx - 1];        // mid-left
    neighbors_arr[4] = frame_data[idx + 1];        // mid-right
    neighbors_arr[5] = frame_data[idx + cols - 1]; // bottom-left
    neighbors_arr[6] = frame_data[idx + cols];     // bottom-center
    neighbors_arr[7] = frame_data[idx + cols + 1]; // bottom-right

    return;
}


// TODO: convert to vector
// ! for testing purposes
#ifdef NAIVE
uint8_t apply_sobel_gradient(uint8_t* neighbors) {
    const int8_t Gx_matrix[8] = {-1, 0, 1, -2, 2, -1, 0, 1};
    const int8_t Gy_matrix[8] = {1, 2, 1, 0, 0, -1, -2, -1};

    int16_t Gx = 0;
    int16_t Gy = 0;

    for (int i = 0; i < 8; ++i) {
        Gx += neighbors[i] * Gx_matrix[i];
        Gy += neighbors[i] * Gy_matrix[i];
    }

    int16_t G = std::abs(Gx) + std::abs(Gy);

    return (G > 255) ? 255 : G;
}
#endif
uint8_t apply_sobel_gradient(uint8_t* neighbors) {
    const int8x8_t Gx_matrix = {-1, 0, 1, -2, 2, -1, 0, 1};
    const int8x8_t Gy_matrix = {1, 2, 1, 0, 0, -1, -2, -1};
    
    // int8x8_t neighbors_vec = vreinterpret_s8_u8(vld1_u8(neighbors));

    // // MAC
    // int8x8_t Gx_accum = vmul_s8(neighbors_vec, Gx_matrix);
    // int8x8_t Gy_accum = vmul_s8(neighbors_vec, Gy_matrix);
    
    // // reduce to scalar
    // int16_t Gx = vaddlv_s8(Gx_accum);
    // int16_t Gy = vaddlv_s8(Gy_accum);

    // int16_t G = std::abs(Gx) + std::abs(Gy);

    // return (G > 255) ? 255 : G;
    return 0;
}

void to442_sobel(Mat& frame, Mat& end_frame) {
    uint8_t neighbors[8];
    for (int row = 1; row < frame.rows - 1; row++) {
        for (int col = 1; col < frame.cols - 1; col++) {
            get_neighbors(frame, row, col, neighbors);
            end_frame.at<uint8_t>(row, col) = apply_sobel_gradient(neighbors);
        }
    }

    return;
}

