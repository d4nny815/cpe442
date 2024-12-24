#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "sobel.hpp"

int main(int argc, char** argv) {
    if (argc != ARG_LEN) {
        printf("Usage: %s <image_path>\n", argv[0]);
        exit(1);
    }

    VideoCapture vid(argv[1]);
    if (!vid.isOpened()) {
        printf("Error: Could not open or find the video.\n");
        exit(1);
    }

    // real time fps
    #ifdef RT_FPS 
    double fps = 0.0;
    double prev_tick = cv::getTickCount();
    #endif

    // avg fps
    size_t frame_count = 0;
    auto start = std::chrono::high_resolution_clock::now();

    int cols = vid.get(CAP_PROP_FRAME_WIDTH);
    int rows = vid.get(CAP_PROP_FRAME_HEIGHT);

    Mat grey_frame(rows, cols, CV_8UC1);
    Mat sobel_frame(rows, cols, CV_8UC1);

    while (1) {
        Mat frame;
        bool succ = vid.read(frame);
        if (!succ) {
            // printf("End of video\n");
            break;
        }

        #ifdef RT_FPS 
        double current_tick = cv::getTickCount();
        double time_elapsed = (current_tick - prev_tick) / cv::getTickFrequency();
        prev_tick = current_tick;
        fps = 1.0 / time_elapsed;
        printf("FPS: %.2f\n", fps); 
        #endif
        frame_count++;

        // TODO: add in here
        to442_greyscale(frame, grey_frame);
        to442_sobel(grey_frame, sobel_frame);


        // namedWindow("Original", WINDOW_NORMAL);
        // resizeWindow("Original", WINDOW_LENGTH, WINDOW_HEIGHT);
        // imshow("Original", frame);
        
        // namedWindow("Greyscale", WINDOW_NORMAL);
        // resizeWindow("Greyscale", WINDOW_LENGTH, WINDOW_HEIGHT);
        // imshow("Greyscale", grey_frame); 
    
        namedWindow("Sobel", WINDOW_NORMAL);
        resizeWindow("Sobel", WINDOW_LENGTH, WINDOW_HEIGHT);
        imshow("Sobel", sobel_frame); 

        if (waitKey(1) == 27)break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double avg_fps = frame_count * 1000.0 / time_elapsed;

    printf("avg fps: %f\n", avg_fps);

    return 0;
}


