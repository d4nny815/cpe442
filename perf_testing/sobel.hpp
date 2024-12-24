#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdint.h>

using namespace cv;
using namespace std;

#define ARG_LEN         (2)
#define WINDOW_LENGTH   (720)
#define WINDOW_HEIGHT   (480)


#define BYTES_PER_PIXEL (3)
#define RED_WEIGHT      (.299f)
#define GREEN_WEIGHT    (.587f)
#define BLUE_WEIGHT     (.114f)

void to442_greyscale(Mat& frame, Mat& end_frame);
uint8_t apply_greyscale(uint8_t* pixel);

void to442_sobel(Mat& frame, Mat& end_frame);
uint8_t apply_sobel_gradient(uint8_t* neighbors);

void get_neighbors(Mat& frame, int row, int col, uint8_t* neighbors_arr);


#endif