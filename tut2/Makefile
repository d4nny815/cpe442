CC = g++

CFLAGS = -Wall 

OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

SRC = image_view.cpp

TARGET = image_viewer

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)