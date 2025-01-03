// sobel.cpp
#include "sobel.hpp"

using namespace cv;
using namespace std;

void to442_greyscale(Mat& frame, Mat& end_frame, int id, size_t partition_size) {
    int start_ind = id * partition_size;
    int end_ind = start_ind + partition_size;

    // q0.8x8 format
    static uint8x8_t red_weights = vdup_n_u8(RED_FIXED);
    static uint8x8_t green_weights = vdup_n_u8(GREEN_FIXED);
    static uint8x8_t blue_weights = vdup_n_u8(BLUE_FIXED);

    for (int row = start_ind; row < end_ind; row++) {
        uint8_t* frame_row = frame.ptr<uint8_t>(row);
        uint8_t* grey_row = end_frame.ptr<uint8_t>(row);
        
        for (int col = 0; col < frame.cols; col += STRIDE) {
            uint8x16x3_t bgr = vld3q_u8(frame_row + col * BYTES_PER_PIXEL);                        

            uint16x8_t grey_u16_low, grey_u16_high;
            // compute greyscale val 
            grey_u16_low = vmull_u8(vget_low_u8(bgr.val[0]), red_weights);
            grey_u16_low = vmlal_u8(grey_u16_low, vget_low_u8(bgr.val[1]), green_weights);
            grey_u16_low = vmlal_u8(grey_u16_low, vget_low_u8(bgr.val[2]), blue_weights);

            grey_u16_high = vmull_u8(vget_high_u8(bgr.val[0]), red_weights);
            grey_u16_high = vmlal_u8(grey_u16_high, vget_high_u8(bgr.val[1]), green_weights);
            grey_u16_high = vmlal_u8(grey_u16_high, vget_high_u8(bgr.val[2]), blue_weights);


            // get the integer part from the values
            uint8x8_t grey_u8_low = vshrn_n_u16(grey_u16_low, 8);
            uint8x8_t grey_u8_high = vshrn_n_u16(grey_u16_high, 8);

            vst1q_u8(grey_row + col, vcombine_u8(grey_u8_low, grey_u8_high));
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


uint8_t apply_sobel_gradient(uint8_t* neighbors) {
    const int8x8_t Gx_matrix = {-1, 0, 1, -2, 2, -1, 0, 1};
    const int8x8_t Gy_matrix = {1, 2, 1, 0, 0, -1, -2, -1};
    
    int8x8_t neighbors_vec = vreinterpret_s8_u8(vld1_u8(neighbors));

    // MAC
    int8x8_t Gx_accum = vmul_s8(neighbors_vec, Gx_matrix);
    int8x8_t Gy_accum = vmul_s8(neighbors_vec, Gy_matrix);
    
    // reduce to scalar
    int16_t Gx = vaddlv_s8(Gx_accum);
    int16_t Gy = vaddlv_s8(Gy_accum);

    int16_t G = abs(Gx) + abs(Gy);

    return (G > MAX_8BIT) ? MAX_8BIT : G;
}


void to442_sobel(Mat& frame, Mat& end_frame, int id, size_t partition_size) {
    int start_ind = id * partition_size;
    int end_ind = start_ind + partition_size;

    if (id == 0) {
        start_ind++; 
    } 
    if (id == NUM_THREADS - 1) {
        end_ind--;
    }
    
    uint8_t neighbors[8];
    for (int row = start_ind; row < end_ind; row++) {
        for (int col = 1; col < frame.cols - 1; col++) {
            get_neighbors(frame, row, col, neighbors);
            end_frame.at<uint8_t>(row, col) = apply_sobel_gradient(neighbors);
        }
    }

    return;
}

