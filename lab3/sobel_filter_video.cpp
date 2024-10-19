#include <opencv2/opencv.hpp>
#include <stdio.h>

#define ARG_LEN         (2)
#define WINDOW_LENGTH   (720)
#define WINDOW_HEIGHT   (480)

using namespace cv;

Mat to442_greyscale(Mat& frame);
uint8_t convert_pixel_to_greyscale(Vec3b pixel);

Mat to442_sobel(Mat& frame);
uint8_t apply_sobel_gradient(uint8_t* neighbors);

void get_neighbors(Mat& frame, int row, int col, uint8_t neighsbors[9]);


int main(int argc, char** argv) {
    if (argc != ARG_LEN) {
        printf("Usage: %s <image_path>\n", argv[0]);
        exit(1);
    }

    printf("attempting to open %s\n", argv[1]);
    VideoCapture vid(argv[1]);
    if (!vid.isOpened()) {
        printf("Error: Could not open or find the video.\n");
        exit(1);
    }

    double fps = vid.get(CAP_PROP_FPS);
    printf("FPS: %lf\n", fps);


    while (1) {
        Mat frame;
        bool succ = vid.read(frame);
        if (!succ) {
            printf("End of video\n");
            break;
        }

        Mat grey_image = to442_greyscale(frame);
        Mat sobel_image = to442_sobel(grey_image);

        //namedWindow("Original", WINDOW_NORMAL);
        //resizeWindow("Original", WINDOW_LENGTH, WINDOW_HEIGHT);
        //imshow("Original", frame);
        
        //namedWindow("Greyscale", WINDOW_NORMAL);
        //resizeWindow("Greyscale", WINDOW_LENGTH, WINDOW_HEIGHT);
        //imshow("Greyscale", grey_image); 
    
        namedWindow("Sobel", WINDOW_NORMAL);
        resizeWindow("Sobel", WINDOW_LENGTH, WINDOW_HEIGHT);
        imshow("Sobel", sobel_image); 

        if (waitKey(1) == 27)break;
    }
    return 0;
}


Mat to442_greyscale(Mat& frame) {
    Mat greyscale(frame.rows, frame.cols, CV_8UC1);

    for (int row = 0; row < frame.rows; row++) {
        for (int col = 0; col < frame.cols; col++) {
            Vec3b pixel = frame.at<Vec3b>(row, col);
            greyscale.at<uint8_t>(row, col) = convert_pixel_to_greyscale(pixel);
        }
    }

    return greyscale;
}


uint8_t convert_pixel_to_greyscale(Vec3b pixel) {
    // CCIR 601 
    // Y = .299 * R + .587 * G +.114 * B
    return .299 * pixel[2] + .587 * pixel[1] + .114 * pixel[0];
}


Mat to442_sobel(Mat& frame) {
    Mat sobel_frame(frame.rows, frame.cols, CV_8UC1);

    for (int row = 1; row < frame.rows - 1; row++) {
        for (int col = 1; col < frame.cols - 1; col++) {
            uint8_t neighbors[9];
            // Mat neighbors = get_neighbors(frame, row, col);
            get_neighbors(frame, row, col, neighbors);
            sobel_frame.at<uint8_t>(row, col) = apply_sobel_gradient(neighbors);
        }
    }

    return sobel_frame;
}


void get_neighbors(Mat& frame, int row, int col, uint8_t neighbors[9]) {
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


uint8_t apply_sobel_gradient(uint8_t* neighbors) {
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
    for (int row = 0; row < 3; row += 2) {
        for (int col = 0; col < 3; col += 2) {
            pixel_value = neighbors[row * 3 + col];
            Gx += pixel_value * Gx_matrix[row][col]; 
            Gy += pixel_value * Gy_matrix[row][col]; 
        }
    }

    int16_t G = std::abs(Gx) + std::abs(Gy);
    return G > 255 ? 255 : G;
}

