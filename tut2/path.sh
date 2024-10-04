#!/bin/bash
echo "USAGE: source path.sh"
CV_PATH=$(find /usr/ -name "opencv4.pc")
echo $CV_PATH
export PKG_CONFIG_PATH=$CV_PATH:$PKG_CONFIG_PATH
