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


// Everything to deal with threading
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

        if (waitKey(1) == 27) break;
    }
    return 0;
}

/**
 * @brief The function that each thread runs to convert an image to greyscale and apply the Sobel filter.
 */
void* thread_sobelfilter_func(void* threadArg) {
    threadArgs_t* args = (threadArgs_t*) threadArg;
    int partition_size = args->frame.rows / NUM_THREADS;

    to442_greyscale(args->frame, args->grey_frame, args->id, partition_size);
    pthread_barrier_wait(&barrier);
    to442_sobel(args->grey_frame, args->sobel_frame, args->id, partition_size);

    pthread_exit(NULL);
}


/**
 * @brief Converts an image to greyscale.
 * @param frame The original color image.
 * @param end_frame The resulting greyscale image.
 * @param id The thread ID, used to determine the partition of the image to process.
 * @param partition_size The size of the partition(image) to process.
 */
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


/**
 * @brief Applies the Sobel filter to a greyscale image.
 * @param frame The original image.
 * @param end_frame The resulting image after applying the Sobel filter.
 * @param id The thread ID, used to determine the partition of the image to process.
 * @param partition_size The size of the partition to process.
 */
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


/**
 * @brief Converts a pixel to greyscale.
 * @param pixel The pixel to convert.
 * @return The greyscale value of the pixel.
 */
uint8_t convert_pixel_to_greyscale(Vec3b pixel) {
    // CCIR 601 
    // Y = .299 * R + .587 * G + .114 * B
    return .299 * pixel[2] + .587 * pixel[1] + .114 * pixel[0];
}


// TODO: update comments
/**
 * @brief Applies the Sobel filter to a 3x3 matrix of pixels.
 * @param neighbors The 3x3 matrix of pixels.
 * @return The gradient value of the pixel.
 */
uint8_t apply_sobel_gradient(uint8_t* neighbors) {
    // Sobel gradient kernels
    const int8_t Gx_matrix[8] = {-1, 0, 1, -2, 2, -1, 0, 1};
    const int8_t Gy_matrix[8] = {1, 2, 1, 0, 0, -1, -2, -1};

    // convert to vectors
    static int8x8_t Gx_mat = vld1_s8(Gx_matrix);
    static int8x8_t Gy_mat = vld1_s8(Gy_matrix);
    int8x8_t neighbors_vec = vreinterpret_s8_u8(vld1_u8(neighbors));

    // MAC
    int8x8_t Gx_accum = vmul_s8(neighbors_vec, Gx_mat);
    int8x8_t Gy_accum = vmul_s8(neighbors_vec, Gy_mat);
    
    // reduce to scalar
    int16_t Gx = vaddlv_s8(Gx_accum);
    int16_t Gy = vaddlv_s8(Gy_accum);

    int16_t G = std::abs(Gx) + std::abs(Gy);

    return (G > 255) ? 255 : G;
}

// TODO: update comments
/**
 * @brief Gets the 3x3 matrix of pixels surrounding a pixel.
 * @param frame The image.
 * @param row The row of the pixel.
 * @param col The column of the pixel.
 * @param neighbors The 3x3 matrix of pixels to fill.
 */
void get_neighbors(Mat& frame, int row, int col, uint8_t neighbors[8]) {
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
