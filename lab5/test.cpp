#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;

#define WINDOW_LENGTH   (720)
#define WINDOW_HEIGHT   (480)
#define TEST_FILE ("../pics/lion.bmp")
#define IMG_TOLERANCE   (5.0)

Mat naive_sobel(Mat& img);
Mat naive_sobel2(Mat& img);
Mat vector_sobel(Mat& img);
int same_images(const Mat& img1, const Mat& img2);


int main(void) {
    Mat img = imread(TEST_FILE, IMREAD_COLOR);

    if (img.empty()) {
        fprintf(stderr, "Error: Could not open or find the image!\n");
        return 1;
    }

    Mat naive = naive_sobel(img);
    Mat naive2 = naive_sobel2(img);
    Mat vec = vector_sobel(img);


    if (!same_images(naive2, vec)) {
        fprintf(stderr, "Error: Images are not identical! Greater than Tolerance\n");
        exit(1);
    }

    printf("Images are identical\n");

    exit(0);
}


int same_images(const Mat& img1, const Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        fprintf(stderr, "Error: Images are not identical (different sizes or types).\n");
        return 1;
    }

    Mat diff;
    absdiff(img1, img2, diff);

    double max_diff = norm(diff, NORM_INF);

    if (max_diff > IMG_TOLERANCE) {
        return 0;
    }

    return 1;
}

#define RED_WEIGHT  (.299)
#define GREEN_WEIGHT  (.587)
#define BLUE_WEIGHT  (.114)
uint8_t greyscale_weights(uint8_t* pixel) {
    return RED_WEIGHT * pixel[2] + GREEN_WEIGHT * pixel[1] + BLUE_WEIGHT * pixel[0];
}

Mat to_greyscale_naive(Mat& frame) {
    Mat greyscale(frame.rows, frame.cols, CV_8UC1);
    
    for (int row = 0; row < frame.rows; row++) {
        uint8_t* frame_row = frame.ptr<uint8_t>(row); // will be 3 times greater than grey
        uint8_t* grey_row = greyscale.ptr<uint8_t>(row);

        for (int col = 0; col <frame.cols; col ++) {
            grey_row[col] = greyscale_weights(&frame_row[col * 3]); 
        }

    }

    return greyscale;
}


#define ROUNDDOWN_16(x) ((x&(~0xf)))
Mat to_greyscale_vec(Mat& frame) {
    Mat greyscale(frame.rows, frame.cols, CV_8UC1);

    for (int row = 0; row < frame.rows; row++) {
        uint8_t* frame_row = frame.ptr<uint8_t>(row); // will be 3 times greater than grey
        uint8_t* grey_row = greyscale.ptr<uint8_t>(row);

        // cant assume cols will be divisible by 16
        int col;
        for (col = 0; col < ROUNDDOWN_16(frame.cols); col ++) {   
            // load 16 3-elem vectors of type u8
            // loads 48 Bytes
            uint8x16x3_t bgr = vld3q_u8(frame_row + col * 3);

            // convert to color vectors
            // blue, green, red vectors (float32 x 4)
            
            const float32x4_t red_weights = vdupq_n_f32(RED_WEIGHT);
            const float32x4_t green_weights = vdupq_n_f32(GREEN_WEIGHT);
            const float32x4_t blue_weights = vdupq_n_f32(BLUE_WEIGHT);
            int i;

            uint32x4_t blues_u32[4], greens_u32[4], reds_u32[4];
            for (i = 0; i < 4; i++) {
                
                blues_u32[i] = vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(bgr.val[0]))));
                greens_u32[i] = vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(bgr.val[1]))));
                reds_u32[i] = vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(bgr.val[2]))));
            }

            float32x4_t blues_f32[4], greens_f32[4], reds_f32[4];
            for (i = 0; i < 4; i++) {
                blues_f32[i] = vcvtq_f32_u32(blues_u32[i]);
                greens_f32[i] = vcvtq_f32_u32(greens_u32[i]);
                reds_f32[i] = vcvtq_f32_u32(reds_u32[i]);
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


            uint8x8_t low = vmovn_u16(greys_u16[0]);  // Convert low part (16 values) to uint8x8
            uint8x8_t high = vmovn_u16(greys_u16[1]);
            uint8x16_t grey = vcombine_u8(low, high);

            vst1q_u8(grey_row + col, grey);
        }
        // remaining cols
        // for (col = 0; col < frame.cols; col++) {
        for (; col < frame.cols; col++) {
            grey_row[col] = greyscale_weights(&frame_row[col * 3]); 
        }
    }

    return greyscale;
}


// =================================================================================

void get_neighbors_9(Mat& frame, int row, int col, uint8_t neighbors[9]) {
    uint8_t* frame_data = frame.ptr<uint8_t>();
    int cols = frame.cols;
    int idx = row * cols + col;

    neighbors[0] = frame_data[idx - cols - 1]; // top-left
    neighbors[1] = frame_data[idx - cols];     // top-center
    neighbors[2] = frame_data[idx - cols + 1]; // top-right
    neighbors[3] = frame_data[idx - 1];        // mid-left
    neighbors[4] = frame_data[idx];            // mid-center (current pixel)
    neighbors[5] = frame_data[idx + 1];        // mid-right
    neighbors[6] = frame_data[idx + cols - 1]; // bottom-left
    neighbors[7] = frame_data[idx + cols];     // bottom-center
    neighbors[8] = frame_data[idx + cols + 1]; // bottom-right
}


uint8_t apply_sobel_naive(uint8_t* neighbors) {
    const int8_t Gx_matrix[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    const int8_t Gy_matrix[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    int16_t Gx = 0;
    int16_t Gy = 0;

    uint8_t pixel_value;
    // skip 2 since centers are 0;
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            pixel_value = neighbors[row * 3 + col];
            Gx += pixel_value * Gx_matrix[row][col]; 
            Gy += pixel_value * Gy_matrix[row][col]; 
        }
    }

    int16_t G = std::abs(Gx) + std::abs(Gy);
    return G > 255 ? 255 : G;
}


Mat to_sobel_naive(Mat& frame) {
    Mat sobel_frame(frame.rows, frame.cols, CV_8UC1);

    for (int row = 1; row < frame.rows - 1; row++) {
        for (int col = 1; col < frame.cols - 1; col++) {
            uint8_t neighbors[9];
            get_neighbors_9(frame, row, col, neighbors);
            sobel_frame.at<uint8_t>(row, col) = apply_sobel_naive(neighbors);
        }
    }

    return sobel_frame;
}


Mat naive_sobel(Mat& img) {
    Mat grey_image = to_greyscale_naive(img);
    Mat sobel_image = to_sobel_naive(grey_image);

    return sobel_image;
}
// =================================================================================

void get_neighbors_8(Mat& frame, int row, int col, uint8_t neighbors[8]) {
    uint8_t* frame_data = frame.ptr<uint8_t>();
    int cols = frame.cols;
    int idx = row * cols + col;

    neighbors[0] = frame_data[idx - cols - 1]; // top-left
    neighbors[1] = frame_data[idx - cols];     // top-center
    neighbors[2] = frame_data[idx - cols + 1]; // top-right
    neighbors[3] = frame_data[idx - 1];        // mid-left
    neighbors[4] = frame_data[idx + 1];        // mid-right
    neighbors[5] = frame_data[idx + cols - 1]; // bottom-left
    neighbors[6] = frame_data[idx + cols];     // bottom-center
    neighbors[7] = frame_data[idx + cols + 1]; // bottom-right
}


uint8_t apply_sobel_naive2(uint8_t* neighbors) {
    // Sobel gradient kernels
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

Mat to_sobel_naive2(Mat& frame) {
    Mat sobel_frame(frame.rows, frame.cols, CV_8UC1);

    for (int row = 1; row < frame.rows - 1; row++) {
        for (int col = 1; col < frame.cols - 1; col++) {
            uint8_t neighbors[8];
            get_neighbors_8(frame, row, col, neighbors);
            sobel_frame.at<uint8_t>(row, col) = apply_sobel_naive2(neighbors);
        }
    }

    return sobel_frame;
}


Mat naive_sobel2(Mat& img) {
    Mat grey_image = to_greyscale_naive(img);
    Mat sobel_image = to_sobel_naive2(grey_image);

    return sobel_image;
}

// =================================================================================

uint8_t apply_sobel_vec(uint8_t* neighbors) {
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

    return (G > 255) ? 255 : G;
}

Mat to_sobel_vec(Mat& frame) {
    Mat sobel_frame(frame.rows, frame.cols, CV_8UC1);

    for (int row = 1; row < frame.rows - 1; row++) {
        for (int col = 1; col < frame.cols - 1; col++) {
            uint8_t neighbors[8];
            get_neighbors_8(frame, row, col, neighbors);
            sobel_frame.at<uint8_t>(row, col) = apply_sobel_vec(neighbors);
        }
    }

    return sobel_frame;
}

Mat vector_sobel(Mat& img) {
    Mat grey_image = to_greyscale_vec(img);
    Mat sobel_image = to_sobel_vec(grey_image);

    return sobel_image;
}
