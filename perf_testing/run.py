'''
    make the program
    exec the program
    when program is done, it will print out the avg fps 
        Format: "avg fps: {avg_fps}"
    save the result to the file
'''

import os
import sys
import subprocess

PROG_NAME = "sobel_filter_video"
VID_PATH = "../videos/"
VID_NAME = "mordetwi480p.mp4"


MAKE_CMDS = [
    ["make", "clean"],
    ["make"],
    # ["make", "naive"],
    # ["make", "mt"],
    # ["make", "vec_sobel"],
    # ["make", "vec_grey"],
]

def make_program():
    for cmd in MAKE_CMDS:
        print(" ".join(cmd))
        subprocess.run(cmd)

def exec_program():
    cmd = f"./{PROG_NAME} {VID_PATH}{VID_NAME}"
    subprocess.run(cmd, shell=True)

def main():
    make_program()
    exec_program()
    
    
    return


if __name__ == "__main__":
    main()