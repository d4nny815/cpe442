import os
import sys
import subprocess

PROG_NAME = "sobel_filter_video"
VID_PATH = "../videos/"
VID_NAME = "mordetwi480p.mp4"
VID_NAME2 = "mordetwi1080p.mp4"
PERF_FILE = "perf.csv"
RUNS = 25

MAKE_CLEAN = ["make", "clean"]
MAKE_CMDS = [
    ["make"],
    ["make", "SOBEL=1"],
    ["make", "SOBEL=1", "FLOAT=1"],
    ["make", "SOBEL=1", "FIXED=1"],

]

TYPES = [
    "NAIVE",
    "SOBEL_VEC",
    "SOBEL_VEC + FLOAT_VEC",
    "SOBEL_VEC + FIXED_VEC",
]

def make_program(i):
    subprocess.run(MAKE_CLEAN)
    subprocess.run(MAKE_CMDS[i])

def exec_program(i):
    vid = ""
    if i:
        vid = VID_NAME
    else:
        vid = VID_NAME2
    cmd = f"./{PROG_NAME} {VID_PATH}{vid}"
    print(cmd)
    return subprocess.run(cmd, shell=True, capture_output=True)

def main():
    file = open(PERF_FILE, "a")
    # file.write("CMD, FPS\n")
    
    for i in range(len(MAKE_CMDS)):
        make_program(i)
        for _ in range(RUNS):
            result = exec_program(True)
            string = result.stdout.decode("utf-8")

            string = string.split(',')
            file.write(f"{TYPES[i]}, {string[0]}, 480p, {string[1]}")
    
    for i in range(len(MAKE_CMDS)):
        make_program(i)
        for _ in range(RUNS):
            result = exec_program(False)
            string = result.stdout.decode("utf-8")

            string = string.split(',')
            file.write(f"{TYPES[i]}, {string[0]}, 1080p, {string[1]}")
    


    file.close()
    return


if __name__ == "__main__":
    main()