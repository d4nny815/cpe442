#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;

#define WINDOW_LENGTH   (720)
#define WINDOW_HEIGHT   (480)
#define TEST_FILE ("../pics/lion.bmp")

Mat naive_sobel(Mat& img);
Mat naive_sobel2(Mat& img);
Mat vector_sobel(Mat& img);


int main(void) {
    Mat img = imread(TEST_FILE, IMREAD_COLOR);

    if (img.empty()) {
        fprintf(stderr, "Error: Could not open or find the image!\n");
        return 1;
    }

    // Mat naive = naive_sobel(img);
    // Mat naive2 = naive_sobel2(img);
    Mat naive = naive_sobel2(img);
    Mat vec = vector_sobel(img);


    // Compare the images
    if (naive.size() != vec.size() || naive.type() != vec.type()) {
        fprintf(stderr, "Images are not identical (different sizes or types).");
        exit(1);
    }
    
    Mat diff;
    absdiff(naive, vec, diff);

    // namedWindow("Difference", WINDOW_NORMAL);
    // resizeWindow("Difference", WINDOW_LENGTH, WINDOW_HEIGHT);
    // imshow("Difference", diff);

    // namedWindow("Difference1", WINDOW_NORMAL);
    // resizeWindow("Difference1", WINDOW_LENGTH, WINDOW_HEIGHT);
    // imshow("Difference1", vec);
    // waitKey(0);


    Mat gray_diff;
    if (diff.channels() > 1)
        cvtColor(diff, gray_diff, COLOR_BGR2GRAY);
    else
        gray_diff = diff;

    // Find the maximum difference
    double max_diff = norm(gray_diff, NORM_INF);

    if (max_diff != 0) {
        fprintf(stderr, "Images are not identical!\n");
        exit(1);
    }

    printf("Images are the same\n");

    return 0;
}


Mat to442_greyscale(Mat& frame) {
    Mat greyscale(frame.rows, frame.cols, CV_8UC1);

    for (int row = 0; row < frame.rows; row++) {
        for (int col = 0; col < frame.cols; col++) {
            Vec3b pixel = frame.at<Vec3b>(row, col);
            greyscale.at<uint8_t>(row, col) = 
                .299 * pixel[2] + .587 * pixel[1] + .114 * pixel[0];
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
    Mat grey_image = to442_greyscale(img);
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

    int8_t Gxs[8];
    int8_t Gys[8];

    for (int i = 0; i < 8; ++i) {
        Gxs[i] = neighbors[i] * Gx_matrix[i];
        Gys[i] = neighbors[i] * Gy_matrix[i];
    }

    for (int i = 0; i < 8; ++i) {
        // printf("i = %d Gx = %hhd Gy = %hhd\n", i, Gxs[i], Gys[i]);
        Gx += Gxs[i];
        Gy += Gys[i];
    }

    // Compute Gx and Gy
    // for (int i = 0; i < 8; ++i) {
    //     printf("i = %d Gx = %hhd Gy = %hhd\n", i, Gx_matrix[i], Gy_matrix[i]);
    //     Gx += neighbors[i] * Gx_matrix[i];
    //     Gy += neighbors[i] * Gy_matrix[i];
    // }

    int16_t G = std::abs(Gx) + std::abs(Gy);
    // printf("Gx = %hhd Gy = %hhd G = %hhu \n",Gx, Gy, G);
    // exit(1);

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
    Mat grey_image = to442_greyscale(img);
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
    Mat grey_image = to442_greyscale(img);
    Mat sobel_image = to_sobel_vec(grey_image);

    return sobel_image;
}

