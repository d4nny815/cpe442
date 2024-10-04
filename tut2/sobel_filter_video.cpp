#include <opencv2/opencv.hpp>
#include <stdio.h>


#define ARG_LEN         (2)
#define WINDOW_LENGTH   (480)
#define WINDOW_HEIGHT   (360)


using namespace cv;

Mat convert_frame_to_greyscale(Mat& frame);
uint8_t convert_pixel_to_greyscale(Vec3b pixel);
Mat apply_sobel_to_greyscale(Mat& frame);
uint8_t apply_sobel_gradient(Mat& neighbors);
Mat get_neighbors(Mat& frame, int row, int col);



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

        Mat grey_image = convert_frame_to_greyscale(frame);
        Mat sobel_image = apply_sobel_to_greyscale(grey_image);

        namedWindow("Original", WINDOW_NORMAL);
        resizeWindow("Original", WINDOW_LENGTH, WINDOW_HEIGHT);
        imshow("Original", frame);
        
        namedWindow("Greyscale", WINDOW_NORMAL);
        resizeWindow("Greyscale", WINDOW_LENGTH, WINDOW_HEIGHT);
        imshow("Greyscale", grey_image); 
    
        namedWindow("Sobel", WINDOW_NORMAL);
        resizeWindow("Sobel", WINDOW_LENGTH, WINDOW_HEIGHT);
        imshow("Sobel", sobel_image); 

        if (waitKey(1) > 0) break;
    }
    return 0;
}


Mat convert_frame_to_greyscale(Mat& frame) {
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

Mat apply_sobel_to_greyscale(Mat& frame) {
    Mat sobel_frame(frame.rows, frame.cols, CV_8UC1);

    for (int row = 0; row < frame.rows; row++) {
        for (int col = 0; col < frame.cols; col++) {
            Mat neighbors = get_neighbors(frame, row, col);
            sobel_frame.at<uint8_t>(row, col) = apply_sobel_gradient(neighbors);
        }
    }

    return sobel_frame;
}

Mat get_neighbors(Mat& frame, int row, int col) {
    int start_row = std::max(0, row - 1);
    int end_row = std::min(frame.rows - 1, row + 1);
    int start_col = std::max(0, col - 1);
    int end_col = std::min(frame.cols - 1, col + 1);

    Rect roi(start_col, start_row, end_col - start_col + 1, end_row - start_row + 1);
    
    return frame(roi).clone();
}


uint8_t apply_sobel_gradient(Mat& neighbors) {

    const int8_t Gx_kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    const int8_t Gy_kernel[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int16_t Gx = 0;
    int16_t Gy = 0;

    // skip 2 since centers are 0;
    for (int row = 0; row < 3; row += 2) {
        for (int col = 0; col < 3; col += 2) {
            uint8_t pixel_value = neighbors.at<uint8_t>(row, col); 
            Gx += pixel_value * Gx_kernel[row][col]; 
            Gy += pixel_value * Gy_kernel[row][col]; 
        }
    }

    // int16_t G = std::sqrt(Gx * Gx + Gy *Gy);
    int16_t G = std::abs(Gx) + std::abs(Gy);
    return G > 255 ? 255 : G;
}

