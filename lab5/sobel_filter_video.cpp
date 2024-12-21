#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <pthread.h>
#include <chrono>

#define ARG_LEN         (2)
#define WINDOW_LENGTH   (720)
#define WINDOW_HEIGHT   (480)
#define RED_WEIGHT  (.299)
#define GREEN_WEIGHT  (.587)
#define BLUE_WEIGHT  (.114)
#define ROUNDDOWN_16(x) ((x&(~0xf)))


using namespace cv;

// main functions
void to442_greyscale(Mat& frame, Mat& end_frame, int id, int partition_size);
uint8_t greyscale_weights(uint8_t* pixel);

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
    
    // real time fps
    #ifdef RT_FPS 
    double fps = 0.0;
    double prev_tick = cv::getTickCount();
    #endif
    
    // avg fps
    size_t frame_count = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (1) {
        Mat frame;
        bool succ = vid.read(frame);
        if (!succ) {
            printf("End of video\n");
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

    auto end = std::chrono::high_resolution_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double avg_fps = frame_count * 1000.0 / time_elapsed;

    printf("avg fps: %f\n", avg_fps);

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
        uint8_t* frame_row = frame.ptr<uint8_t>(row);
        uint8_t* grey_row = end_frame.ptr<uint8_t>(row);
        for (int col = 0; col < frame.cols; col++) {
            grey_row[col] = greyscale_weights(&frame_row[col * 3]); 
        }
    }

    // for (int row = start_ind; row < end_ind; row++) {
    //     uint8_t* frame_row = frame.ptr<uint8_t>(row); // will be 3 times greater than grey
    //     uint8_t* grey_row = end_frame.ptr<uint8_t>(row);

    //     // cant assume cols will be divisible by 16
    //     int col;
    //     for (col = 0; col < ROUNDDOWN_16(frame.cols); col ++) {   
    //         // load 16 3-elem vectors of type u8
    //         // loads 48 Bytes
    //         uint8x16x3_t bgr = vld3q_u8(frame_row + col * 3);

    //         // convert to color vectors
    //         // blue, green, red vectors (float32 x 4)
            
    //         const float32x4_t red_weights = vdupq_n_f32(RED_WEIGHT);
    //         const float32x4_t green_weights = vdupq_n_f32(GREEN_WEIGHT);
    //         const float32x4_t blue_weights = vdupq_n_f32(BLUE_WEIGHT);
    //         int i;

    //         uint32x4_t blues_u32[4], greens_u32[4], reds_u32[4];
    //         for (i = 0; i < 4; i++) {
                
    //             blues_u32[i] = vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(bgr.val[0]))));
    //             greens_u32[i] = vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(bgr.val[1]))));
    //             reds_u32[i] = vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(bgr.val[2]))));
    //         }

    //         float32x4_t blues_f32[4], greens_f32[4], reds_f32[4];
    //         for (i = 0; i < 4; i++) {
    //             blues_f32[i] = vcvtq_f32_u32(blues_u32[i]);
    //             greens_f32[i] = vcvtq_f32_u32(greens_u32[i]);
    //             reds_f32[i] = vcvtq_f32_u32(reds_u32[i]);
    //         }

    //         // compute greyscale vals
    //         float32x4_t greys_f32[4];
    //         for (i = 0; i < 4; i++) {
    //             blues_f32[i] = vmulq_f32(blues_f32[i], blue_weights);
    //             greens_f32[i] = vmulq_f32(greens_f32[i], green_weights);
    //             reds_f32[i] = vmulq_f32(reds_f32[i], red_weights);

    //             greys_f32[i] = vaddq_f32(vaddq_f32(blues_f32[i], greens_f32[i]), reds_f32[i]);
    //         }

    //         // store the greyscale vals
    //         // 4 f32x4 -> 4 u32x4 -> 2 u16x8 -> 1 u8x16
    //         uint32x4_t greys_u32[4];
    //         for (i = 0; i < 4; i++) {
    //             greys_u32[i] = vcvtq_u32_f32(greys_f32[i]);
    //         }

    //         uint16x8_t greys_u16[2];
    //         for (i = 0; i < 2; i++) {
    //             uint16x4_t low = vmovn_u32(greys_u32[2 * i]);
    //             uint16x4_t high = vmovn_u32(greys_u32[2 * i + 1]);
    //             greys_u16[i] = vcombine_u16(low, high);
    //         }


    //         uint8x8_t low = vmovn_u16(greys_u16[0]);  // Convert low part (16 values) to uint8x8
    //         uint8x8_t high = vmovn_u16(greys_u16[1]);
    //         uint8x16_t grey = vcombine_u8(low, high);

    //         vst1q_u8(grey_row + col, grey);
    //     }
    //     // remaining cols
    //     // for (col = 0; col < frame.cols; col++) {
    //     for (; col < frame.cols; col++) {
    //         grey_row[col] = greyscale_weights(&frame_row[col * 3]); 
    //     }
    // }

    return;
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
uint8_t greyscale_weights(uint8_t* pixel) {
    return RED_WEIGHT * pixel[2] + GREEN_WEIGHT * pixel[1] + BLUE_WEIGHT * pixel[0];
}


/**
 * @brief Applies the Sobel filter to a pixel.
 * @param neighbors The neighbors of the pixel. Not including the pixel itself.
 * @return The Sobel gradient of the pixel.
 */
uint8_t apply_sobel_gradient(uint8_t* neighbors) {
    // Sobel gradient kernels
    const int8x8_t Gx_matrix = {-1, 0, 1, -2, 2, -1, 0, 1};
    const int8x8_t Gy_matrix = {1, 2, 1, 0, 0, -1, -2, -1};

    // convert to vectors
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

/**
 * @brief Gets the neighbors of a pixel, not including the pixel itself. 
 * @note The neighbors are stored in the following order: Top-left, Top-center, Top-right, Mid-left, Mid-right, Bottom-left, Bottom-center, Bottom-right.
 * @param frame The image.
 * @param row The row of the pixel.
 * @param col The column of the pixel.
 * @param neighbors The array to store the neighbors.
 * @return void
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
