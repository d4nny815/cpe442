CC = g++

CFLAGS = -Wall -O3 -Werror -g -std=c++11 -lpthread -march=armv8-a -mtune=cortex-a72 -ffast-math -fopenmp -findirect-inlining -funroll-loops -floop-parallelize-all -ftree-vectorize -ftree-loop-vectorize


OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

LD_FLAGS = -L/usr/local/lib

RT_FPS_OPTS = -DRT_FPS

SRC = sobel_filter_video

all: $(SRC)

$(SRC): $(SRC).cpp
	$(CC) $(CFLAGS) -o $(SRC) $< $(OPENCV_FLAGS)

rtfps: $(SRC).cpp
	$(CC) $(CFLAGS) $(RT_FPS_OPTS) -o $(SRC) $< $(OPENCV_FLAGS)

test: test.cpp
	$(CC) $(CFLAGS) $(LD_FLAGS) -o $@ $< $(OPENCV_FLAGS)

fun: experimenting.cpp
	$(CC) $(CFLAGS) $(LD_FLAGS) -o $@ $< 

clean:
	rm -f sobel_filter_video test fun
