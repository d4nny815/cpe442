// sobel.hpp
#ifndef SOBEL_HPP
#define SOBEL_HPP

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdint.h>

using namespace cv;
using namespace std;

#define NUM_THREADS     (4)

#define MAX_8BIT        (0xff)
#define STRIDE          (16)
#define BYTES_PER_PIXEL (3)
#define RED_WEIGHT      (.299f)
#define GREEN_WEIGHT    (.587f)
#define BLUE_WEIGHT     (.114f)
#define RED_FIXED       ((uint8_t)(RED_WEIGHT * MAX_8BIT + 1))
#define GREEN_FIXED     ((uint8_t)(GREEN_WEIGHT * MAX_8BIT + 1))
#define BLUE_FIXED      ((uint8_t)(BLUE_WEIGHT * MAX_8BIT + 1))


void to442_greyscale(Mat& frame, Mat& end_frame, int id, size_t partition_size);
uint8_t apply_greyscale(uint8_t* pixel);

void to442_sobel(Mat& frame, Mat& end_frame, int id, size_t partition_size);
uint8_t apply_sobel_gradient(uint8_t* neighbors);

void get_neighbors(Mat& frame, int row, int col, uint8_t* neighbors_arr);


#endif