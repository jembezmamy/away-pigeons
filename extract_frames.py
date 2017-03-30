import subprocess
import sys
import os
import glob

# python3 extract_frames.py input_glob_path output_directory    

input_file_names = sys.argv[1:-2]
output_path = sys.argv[-1]

if not os.path.exists(output_path):
    os.makedirs(output_path)

for i, file_name in enumerate(input_file_names):
    subprocess.call(["ffmpeg", "-i", file_name, "-vf", "fps=1", "{}/{}-%d.png".format(output_path, i)])
    # for j, frame_name in enumerate(glob.glob(output_path + "/tmp/*.png")):
    #     subprocess.call([
    #         "convert", "-resize", "640x384!", "-crop", "64x64",
    #         frame_name, "+repage", "{}/{}-{}-%d.png".format(output_path, i, j)
    #     ])
    #     os.unlink(frame_name)
