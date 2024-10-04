CC = g++

CFLAGS = -Wall -O3 -Werror -g

OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

all: sobel_filter_image sobel_filter_video

sobel_filter_image: sobel_filter_image.cpp
	$(CC) $(CFLAGS) -o $@ $< $(OPENCV_FLAGS)

sobel_filter_video: sobel_filter_video.cpp
	$(CC) $(CFLAGS) -o $@ $< $(OPENCV_FLAGS)


clean:
	rm -f sobel_filter_image sobel_filter_video