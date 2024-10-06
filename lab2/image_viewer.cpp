#include <opencv2/opencv.hpp>
#include <stdio.h>


#define ARG_LEN         (2)
#define WINDOW_LENGTH   (800)
#define WINDOW_HEIGHT   (600)


using namespace cv;


int main(int argc, char** argv) {
    if (argc != ARG_LEN) {
        printf("Usage: %s <image_path>\n", argv[0]);
        exit(1);
    }

    Mat image = imread(argv[1], IMREAD_COLOR);

    if (image.empty()) {
        printf("Error: Could not open or find the image.\n");
        exit(1);
    }

    namedWindow("Image", WINDOW_NORMAL);
    resizeWindow("Image", WINDOW_LENGTH, WINDOW_HEIGHT);
    imshow("Image", image);

    waitKey(0);

    return 0;
}