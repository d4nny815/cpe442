# Makefile
CC = g++

CFLAGS = -Wall -Werror -O0

OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

all: image_viewer 

image_viewer: image_viewer.cpp
	$(CC) $(CFLAGS) -o $@ $< $(OPENCV_FLAGS)

clean:
	rm -f image_viewer