CC = g++

CFLAGS = -Wall -O0 -Werror -g -std=c++11 -lpthread

OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

LD_FLAGS = -L/usr/local/lib

all: sobel_filter_video

sobel_filter_video: sobel_filter_video.cpp
	$(CC) $(CFLAGS) $(LD_FLAGS) -o $@ $< $(OPENCV_FLAGS)

test: test.cpp
	$(CC) $(CFLAGS) $(LD_FLAGS) -o $@ $< $(OPENCV_FLAGS)

fun: experimenting.cpp
	$(CC) $(CFLAGS) $(LD_FLAGS) -o $@ $< 

clean:
	rm -f sobel_filter_video test fun
