#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "sobel.hpp"

#define ARG_LEN         (2)
#define WINDOW_LENGTH   (720)
#define WINDOW_HEIGHT   (480)

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

        threadArgs_t threadArgs[NUM_THREADS];

        for (int i = 0; i < NUM_THREADS; i++) {
            threadArgs[i] = {i, frame, grey_frame, sobel_frame};
            pthread_create(&threads[i], NULL, thread_sobelfilter_func, (void*)&threadArgs[i]);
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }


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

    printf("%f, %d\n", avg_fps, NUM_THREADS);

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

