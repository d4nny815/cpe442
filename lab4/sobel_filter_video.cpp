#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <pthread.h>

#define ARG_LEN         (2)
#define WINDOW_LENGTH   (720)
#define WINDOW_HEIGHT   (480)

using namespace cv;

// main functions
void to442_greyscale(Mat& frame, Mat& end_frame, int id, int partition_size);
uint8_t convert_pixel_to_greyscale(Vec3b pixel);

void to442_sobel(Mat& frame, Mat& end_frame, int id, int partition_size);
uint8_t apply_sobel_gradient(uint8_t* neighbors);

void get_neighbors(Mat& frame, int row, int col, uint8_t neighbors[9]);


// Everything do deal with threading
#define NUM_THREADS     (4)
pthread_t threads[NUM_THREADS];
typedef struct threadArgs_t {
    int id;
    Mat frame;
    Mat grey_frame;
    Mat sobel_frame;
} threadArgs_t;
pthread_barrier_t barrier;
void* thread_sobelfilter_func(void* threadArg);


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

    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    double fps = 0.0;
    double prev_tick = cv::getTickCount();

    while (1) {
        Mat frame;
        bool succ = vid.read(frame);
        if (!succ) {
            printf("End of video\n");
            break;
        }

        double current_tick = cv::getTickCount();
        double time_elapsed = (current_tick - prev_tick) / cv::getTickFrequency();
        prev_tick = current_tick;
        fps = 1.0 / time_elapsed;

        printf("FPS: %.2f\n", fps); 


        threadArgs_t threadArgs[NUM_THREADS];
        Mat grey_frame(frame.rows, frame.cols, CV_8UC1);
        Mat sobel_image(frame.rows, frame.cols, CV_8UC1);

        for (int i = 0; i < NUM_THREADS; i++) {
            threadArgs[i] = {i, frame, grey_frame, sobel_image};
            pthread_create(&threads[i], NULL, thread_sobelfilter_func, (void*)&threadArgs[i]);
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }

        namedWindow("Sobel", WINDOW_NORMAL);
        resizeWindow("Sobel", WINDOW_LENGTH, WINDOW_HEIGHT);
        imshow("Sobel", sobel_image);
        // imshow("Sobel", grey_frame);

        if (waitKey(1) == 27) break;
    }
    return 0;
}


void* thread_sobelfilter_func(void* threadArg) {
    threadArgs_t* args = (threadArgs_t*) threadArg;
    int partition_size = args->frame.rows / NUM_THREADS;

    to442_greyscale(args->frame, args->grey_frame, args->id, partition_size);
    pthread_barrier_wait(&barrier);
    to442_sobel(args->grey_frame, args->sobel_frame, args->id, partition_size);

    pthread_exit(NULL);
}


void to442_greyscale(Mat& frame, Mat& end_frame, int id, int partition_size) {
    int start_ind = id * partition_size;
    int end_ind = start_ind + partition_size;

    for (int row = start_ind; row < end_ind; row++) {
        for (int col = 0; col < frame.cols; col++) {
            Vec3b pixel = frame.at<Vec3b>(row, col);
            end_frame.at<uint8_t>(row, col) = convert_pixel_to_greyscale(pixel);
        }
    }
}


void to442_sobel(Mat& frame, Mat& end_frame, int id, int partition_size) {
    int start_ind = id * partition_size;
    int end_ind = start_ind + partition_size;

    if (id == 0) {
        start_ind++; 
    } 
    if (id == NUM_THREADS - 1) {
        end_ind--;
    }

    uint8_t neighbors[9];
    for (int row = start_ind; row < end_ind; row++) {
        for (int col = 1; col < frame.cols - 1; col++) { 
            get_neighbors(frame, row, col, neighbors);
            end_frame.at<uint8_t>(row, col) = apply_sobel_gradient(neighbors);
        }
    }
}


uint8_t convert_pixel_to_greyscale(Vec3b pixel) {
    // CCIR 601 
    // Y = .299 * R + .587 * G + .114 * B
    return .299 * pixel[2] + .587 * pixel[1] + .114 * pixel[0];
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

    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            uint8_t pixel_value = neighbors[row * 3 + col];
            Gx += pixel_value * Gx_matrix[row][col];
            Gy += pixel_value * Gy_matrix[row][col];
        }
    }

    int16_t G = std::abs(Gx) + std::abs(Gy);
    return G > 255 ? 255 : G;
}

void get_neighbors(Mat& frame, int row, int col, uint8_t neighbors[9]) {
    uint8_t* frame_data = frame.ptr<uint8_t>();
    int cols = frame.cols;
    int idx = row * cols + col;

    neighbors[0] = frame_data[idx - cols - 1]; // top-left
    neighbors[1] = frame_data[idx - cols];     // top-center
    neighbors[2] = frame_data[idx - cols + 1]; // top-right
    neighbors[3] = frame_data[idx - 1];        // mid-left
    neighbors[4] = frame_data[idx];            // cur pixel
    neighbors[5] = frame_data[idx + 1];        // mid-right
    neighbors[6] = frame_data[idx + cols - 1]; // bottom-left
    neighbors[7] = frame_data[idx + cols];     // bottom-center
    neighbors[8] = frame_data[idx + cols + 1]; // bottom-right
}
