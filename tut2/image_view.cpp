#include <opencv2/opencv.hpp>
#include <stdio.h>

#define ARG_LEN         (2)

int main(int argc, char** argv) {
    if (argc != ARG_LEN) {
        printf("Usage: %s <image_path>\n", argv[0]);
        exit(1);
    }

    // Read the image file using C++ style cv::imread function
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Check for failure
    if (image.empty()) {
        printf("Error: Could not open or find the image.\n");
        exit(1);
    }

    // Create a window
    cv::namedWindow("Image Viewer", cv::WINDOW_AUTOSIZE);

    // Show the image inside the window
    cv::imshow("Image Viewer", image);

    // Wait for any keystroke in the window
    cv::waitKey(0);

    return 0;
}




