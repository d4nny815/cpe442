CC = g++

CFLAGS = -Wall -O0 -Werror -g

OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

RT_FPS_OPTS = -DRT_FPS

SRC = sobel_filter_video

all: $(SRC)

$(SRC): $(SRC).cpp
	$(CC) $(CFLAGS) -o $(SRC) $< $(OPENCV_FLAGS)

rtfps: $(SRC).cpp
	$(CC) $(CFLAGS) $(RT_FPS_OPTS) -o $(SRC) $< $(OPENCV_FLAGS)

clean:
	rm -f sobel_filter_video
