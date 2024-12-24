CC = g++

CFLAGS = -Wall -O0 -Werror -g -std=c++11 -lpthread

OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

LD_FLAGS = -L/usr/local/lib

NAIVE_OPTS = -DNAIVE

OPTS = 

all: sobel_filter_video

sobel_filter_video: sobel_filter_video.o sobel.o
	$(CC) $(CFLAGS) $^ $(OPENCV_FLAGS) -o $@

sobel_filter_video.o: sobel_filter_video.cpp
	$(CC) $(CFLAGS) -c $< $(OPENCV_FLAGS) -o $@ 

sobel.o: sobel.cpp sobel.hpp
	$(CC) $(CFLAGS) $(OPTS) -c $< $(OPENCV_FLAGS) -o $@


naive:
	OPTS += $(NAIVE_OPTS)

clean:
	rm -f sobel_filter_video *.o
