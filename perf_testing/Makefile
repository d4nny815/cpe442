CC = g++
CFLAGS = -Wall -O0 -Werror -g -std=c++11 -lpthread
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

BUILD_DIR = build
DUMP_DIR = dump

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

all: dirs sobel_filter_video debug
debug: assembly dump

sobel_filter_video: $(BUILD_DIR)/sobel_filter_video.o $(BUILD_DIR)/sobel.o
	$(CC) $(CFLAGS) $^ $(OPENCV_FLAGS) -o $@

$(BUILD_DIR)/sobel_filter_video.o: sobel_filter_video.cpp dirs
	$(CC) $(CFLAGS) -c $< $(OPENCV_FLAGS) -o $@ 

$(BUILD_DIR)/sobel.o: sobel.cpp sobel.hpp dirs
	$(CC) $(CFLAGS) $(OPTS) -c $< $(OPENCV_FLAGS) -o $@ 

# =====================================================
# this is for debug stuff
# =====================================================

$(DUMP_DIR)/%.s: %.cpp dirs
	$(CC) $(CFLAGS) -S $< $(OPENCV_FLAGS) -o $@

assembly: $(DUMP_DIR)/sobel_filter_video.s $(DUMP_DIR)/sobel.s

$(DUMP_DIR)/sobel_filter_video.dump: sobel_filter_video dirs
	objdump -d -S sobel_filter_video > $@

dump: $(DUMP_DIR)/sobel_filter_video.dump

clean:
	rm -rf sobel_filter_video $(BUILD_DIR) $(DUMP_DIR)

dirs:
	mkdir -p $(BUILD_DIR) $(DUMP_DIR)

.PHONY: clean all dirs dump assembly debug