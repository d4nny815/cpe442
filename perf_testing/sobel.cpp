#include "sobel.hpp"


using namespace cv;
using namespace std;

inline uint8_t apply_greyscale(uint8_t* pixel) {
    return RED_WEIGHT * pixel[2] + GREEN_WEIGHT * pixel[1] + BLUE_WEIGHT * pixel[0];
}

void to442_greyscale(Mat& frame, Mat& end_frame, int id, size_t partition_size) {
    int start_ind = id * partition_size;
    int end_ind = start_ind + partition_size;


#ifdef VEC_GREY_FLOAT
    #define ROUNDDOWN_16(x) ((x&(~0xf)))
    for (int row = start_ind; row < end_ind; row++) {
        uint8_t* frame_row = frame.ptr<uint8_t>(row); // will be 3 times greater than grey
        uint8_t* grey_row = end_frame.ptr<uint8_t>(row);

        // cant assume cols will be divisible by 16 480 yes 1080 no
        int col;
        for (col = 0; col < ROUNDDOWN_16(frame.cols); col ++) {   
            // load 16 3-elem vectors of type u8
            // loads 48 Bytes
            uint8x16x3_t bgr = vld3q_u8(frame_row + col * 3);
            
            static float32x4_t red_weights = vdupq_n_f32(RED_WEIGHT);
            static float32x4_t green_weights = vdupq_n_f32(GREEN_WEIGHT);
            static float32x4_t blue_weights = vdupq_n_f32(BLUE_WEIGHT);
            int i;

            float32x4_t blues_f32[4], greens_f32[4], reds_f32[4];

            // convert everything to floating vectors
            // ? there has to be a better way
            for (i = 0; i < 4; i++) {
                blues_f32[i] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(bgr.val[0])))));
                greens_f32[i] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(bgr.val[1])))));
                reds_f32[i] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(bgr.val[2])))));
            }

            // compute greyscale vals
            float32x4_t greys_f32[4];
            for (i = 0; i < 4; i++) {
                blues_f32[i] = vmulq_f32(blues_f32[i], blue_weights);
                greens_f32[i] = vmulq_f32(greens_f32[i], green_weights);
                reds_f32[i] = vmulq_f32(reds_f32[i], red_weights);

                greys_f32[i] = vaddq_f32(vaddq_f32(blues_f32[i], greens_f32[i]), reds_f32[i]);
            }

            // store the greyscale vals
            // 4 f32x4 -> 4 u32x4 -> 2 u16x8 -> 1 u8x16
            uint32x4_t greys_u32[4];
            for (i = 0; i < 4; i++) {
                greys_u32[i] = vcvtq_u32_f32(greys_f32[i]);
            }

            uint16x8_t greys_u16[2];
            for (i = 0; i < 2; i++) {
                uint16x4_t low = vmovn_u32(greys_u32[2 * i]);
                uint16x4_t high = vmovn_u32(greys_u32[2 * i + 1]);
                greys_u16[i] = vcombine_u16(low, high);
            }

            uint8x8_t low = vmovn_u16(greys_u16[0]);
            uint8x8_t high = vmovn_u16(greys_u16[1]);
            uint8x16_t grey = vcombine_u8(low, high);

            // store the values
            vst1q_u8(grey_row + col, grey);
        }
        // remaining cols
        for (; col < frame.cols; col++) {
            grey_row[col] = apply_greyscale(&frame_row[col * BYTES_PER_PIXEL]); 
        }
    }

    return;
#elif defined(VEC_GREY_FIXED)
    #define STRIDE  (8)

    // q8.8x8 format
    static uint8x8_t red_weights = vdup_n_u8(RED_WEIGHT * 256 + 1);
    static uint8x8_t green_weights = vdup_n_u8(GREEN_WEIGHT * 256 + 1);
    static uint8x8_t blue_weights = vdup_n_u8(BLUE_WEIGHT * 256 + 1);

    for (int row = start_ind; row < end_ind; row++) {
        uint8_t* frame_row = frame.ptr<uint8_t>(row);
        uint8_t* grey_row = end_frame.ptr<uint8_t>(row);
        
        uint16x8_t grey_u16;  
        uint8x8_t grey_u8;

        for (int col = 0; col < frame.cols; col += STRIDE) {
            uint8x8x3_t bgr = vld3_u8(frame_row + col * BYTES_PER_PIXEL);                        

            // compute greyscale val 
            grey_u16 = vmull_u8(bgr.val[0], red_weights);
            grey_u16 = vmlal_u8(grey_u16, bgr.val[1], green_weights);
            grey_u16 = vmlal_u8(grey_u16, bgr.val[2], blue_weights);

            // get the integer part
            grey_u8 = vshrn_n_u16(grey_u16, 8);

            vst1_u8(grey_row + col, grey_u8);
        }
    }

    return;
#else
    for (int row = start_ind; row < end_ind; row++) {
        uint8_t* frame_row = frame.ptr<uint8_t>(row);
        uint8_t* grey_row = end_frame.ptr<uint8_t>(row);
        
        for (int col = 0; col < frame.cols; col++) {
            grey_row[col] = apply_greyscale(&frame_row[col * BYTES_PER_PIXEL]);
        }
    }

    return; 
#endif
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
#ifdef SOBEL
    const int8x8_t Gx_matrix = {-1, 0, 1, -2, 2, -1, 0, 1};
    const int8x8_t Gy_matrix = {1, 2, 1, 0, 0, -1, -2, -1};
    
    int8x8_t neighbors_vec = vreinterpret_s8_u8(vld1_u8(neighbors));

    // MAC
    int8x8_t Gx_accum = vmul_s8(neighbors_vec, Gx_matrix);
    int8x8_t Gy_accum = vmul_s8(neighbors_vec, Gy_matrix);
    
    // reduce to scalar
    int16_t Gx = vaddlv_s8(Gx_accum);
    int16_t Gy = vaddlv_s8(Gy_accum);

    int16_t G = std::abs(Gx) + std::abs(Gy);

    return (G > MAX_8BIT) ? MAX_8BIT : G;
#else
    const int8_t Gx_matrix[8] = {-1, 0, 1, -2, 2, -1, 0, 1};
    const int8_t Gy_matrix[8] = {1, 2, 1, 0, 0, -1, -2, -1};

    int16_t Gx = 0;
    int16_t Gy = 0;

    for (int i = 0; i < 8; ++i) {
        Gx += neighbors[i] * Gx_matrix[i];
        Gy += neighbors[i] * Gy_matrix[i];
    }

    int16_t G = std::abs(Gx) + std::abs(Gy);

    return (G > MAX_8BIT) ? MAX_8BIT : G;
#endif
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

