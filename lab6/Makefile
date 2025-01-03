# Makefile
CC = g++
CFLAGS = -Wall -O3 -Werror -g -std=c++11 
LD_FLAGS = -lpthread
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

RT_FPS_OPTS = -DRT_FPS

# flags added
CFLAGS += -funroll-loops -floop-parallelize-all -floop-interchange  -floop-block


.PHONY: clean all

ifeq ($(RTFPS), 1)
CFLAGS += $(RT_FPS_OPTS)
endif

all: sobel_filter_video

sobel_filter_video: sobel_filter_video.o sobel.o
	$(CC) $(CFLAGS) $^ $(OPENCV_FLAGS) -o $@

sobel_filter_video.o: sobel_filter_video.cpp
	$(CC) $(CFLAGS) $(LD_FLAGS) -c $< $(OPENCV_FLAGS) -o $@ 

sobel.o: sobel.cpp sobel.hpp
	$(CC) $(CFLAGS) -c $< $(OPENCV_FLAGS) -o $@ 

clean:
	rm -rf sobel_filter_video *.o
