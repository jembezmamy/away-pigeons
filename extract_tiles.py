import subprocess
import sys
import os
import glob
from random import randint
import yaml

# python3 extract_tiles.py input_glob_path output_directory

config = yaml.safe_load(open("config.yml"))

input_file_names = sys.argv[1:-2]
output_path = sys.argv[-1]
tile_width = config['training']['patch_width']
tile_height = config['training']['patch_height']

if not os.path.exists(output_path):
    os.makedirs(output_path)

for i, file_name in enumerate(input_file_names):
    (width, height) = [int(x) for x in subprocess.check_output([
        "identify", "-format", "%[w] %[h]",
        file_name
    ]).decode("utf-8").split(" ")]

    x = randint(0, width - tile_width)
    y = randint(0, height - tile_height)

    subprocess.call([
        "convert", "-crop", "{}x{}+{}+{}".format(tile_width, tile_height, x, y),
        file_name, "+repage", "{}/{}.png".format(output_path, i)
    ])
