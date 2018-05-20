import subprocess
import sys
import os
from random import randint
import yaml
import re
import xml.etree.ElementTree
import math

# python3 extract_tiles.py input_glob_path

config = yaml.safe_load(open("config.yml"))

input_file_names = sys.argv[1:-2]
output_path = config['training']['storage_path']
tile_width = config['training']['patch_width']
tile_height = config['training']['patch_height']

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists("{}/positive".format(output_path)):
    os.makedirs("{}/positive".format(output_path))
if not os.path.exists("{}/negative".format(output_path)):
    os.makedirs("{}/negative".format(output_path))

previous_tiles = []

def intersects_with(x, y, other_tiles):
    for tile in other_tiles:
        if abs(tile[0] - x) < tile_width and abs(tile[1] - y) < tile_height:
            return True
    return False

def save_tile(input_file_name, x, y, output_file_name):
    subprocess.call([
        "convert", "-crop", "{}x{}+{}+{}".format(tile_width, tile_height, x - tile_width/2, y - tile_width/2),
        input_file_name, "+repage", output_file_name
    ])

for i, file_name in enumerate(input_file_names):
    xml_file_name = re.sub(r"\.[^.]+$", ".xml", file_name)
    (width, height) = [int(x) for x in subprocess.check_output([
        "identify", "-format", "%[w] %[h]",
        file_name
    ]).decode("utf-8").split(" ")]
    local_tiles = []
    positive_count = 0
    if os.path.exists(xml_file_name):
        pascal_voc = xml.etree.ElementTree.parse(xml_file_name).getroot()
        for rectangle in pascal_voc.iter("object"):
            x = math.floor((int(rectangle.find("*/xmin").text) + int(rectangle.find("*/xmax").text)) / 2)
            y = math.floor((int(rectangle.find("*/ymin").text) + int(rectangle.find("*/ymax").text)) / 2)
            positive_count += 1
            save_tile(file_name, x, y, "{}/positive/{}-{}.png".format(output_path, i, positive_count))
            local_tiles.append([x, y])
            previous_tiles.append([x, y])
    negative_count = 0
    for _ in range(positive_count):
        while True:
            x = randint(0, width - tile_width) + tile_width/2
            y = randint(0, height - tile_height) + tile_height/2
            if not intersects_with(x, y, local_tiles):
                break
        local_tiles.append([x, y])
        negative_count += 1
        save_tile(file_name, x, y, "{}/negative/{}-{}.png".format(output_path, i, negative_count))
    j = 0
    while j < len(previous_tiles):
        tile = previous_tiles[j]
        if not intersects_with(tile[0], tile[1], local_tiles):
            local_tiles.append([x, y])
            negative_count += 1
            save_tile(file_name, tile[0], tile[1], "{}/negative/{}-{}.png".format(output_path, i, negative_count))
            del previous_tiles[j]
        else:
            j += 1
