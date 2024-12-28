CC = g++

CFLAGS = -Wall -O0 -Werror -g -std=c++11 -lpthread

OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

FLOAT_OPT = -DVEC_GREY_FLOAT
FIXED_OPT = -DVEC_GREY_FIXED
SOBEL_OPT = -DSOBEL

OPTS = 

ifeq ($(SOBEL), 1)
	OPTS += $(SOBEL_OPT)
endif
ifeq ($(FIXED), 1)
	OPTS += $(FIXED_OPT)
endif
ifeq ($(FLOAT), 1)
	OPTS += $(FLOAT_OPT)
endif

all: sobel_filter_video

sobel_filter_video: sobel_filter_video.o sobel.o
	$(CC) $(CFLAGS) $^ $(OPENCV_FLAGS) -o $@

sobel_filter_video.o: sobel_filter_video.cpp
	$(CC) $(CFLAGS) -c $< $(OPENCV_FLAGS) -o $@ 

sobel.o: sobel.cpp sobel.hpp
	$(CC) $(CFLAGS) $(OPTS) -c $< -I./ $(OPENCV_FLAGS) -o $@ 

clean:
	rm -f sobel_filter_video *.o
