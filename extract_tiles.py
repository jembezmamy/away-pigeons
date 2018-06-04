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
output_path = "{}/images".format(config['training']['storage_path'])

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists("{}/positive".format(output_path)):
    os.makedirs("{}/positive".format(output_path))
if not os.path.exists("{}/negative".format(output_path)):
    os.makedirs("{}/negative".format(output_path))

previous_tiles = []

def intersects_with(x0, y0, x1, y1, other_tiles):
    for tile in other_tiles:
        if x0 <= tile[2] and x1 >= tile[0] and y0 <= tile[3] and y1 >= tile[1]:
            return True
    return False

def save_tile(input_file_name, x0, y0, x1, y1, output_file_name):
    subprocess.call([
        "convert", "-crop", "{}x{}+{}+{}".format(x1 - x0, y1 - y0, x0, y0),
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
            x0 = int(rectangle.find("*/xmin").text)
            x1 = int(rectangle.find("*/xmax").text)
            y0 = int(rectangle.find("*/ymin").text)
            y1 = int(rectangle.find("*/ymax").text)
            positive_count += 1
            save_tile(file_name, x0, y0, x1, y1, "{}/positive/{}-{}.jpg".format(output_path, i, positive_count))
            local_tiles.append([x0, y0, x1, y1])
            previous_tiles.append([x0, y0, x1, y1])
    negative_count = 0
    for j in range(positive_count):
        while True:
            w = local_tiles[j][2] - local_tiles[j][0]
            h = local_tiles[j][3] - local_tiles[j][1]
            x0 = randint(0, width - w)
            y0 = randint(0, height - h)
            x1 = x0 + w
            y1 = y0 + h
            if not intersects_with(x0, y0, x1, y1, local_tiles):
                break
        local_tiles.append([x0, y0, x1, y1])
        negative_count += 1
        save_tile(file_name, x0, y0, x1, y1, "{}/negative/{}-{}.jpg".format(output_path, i, negative_count))
    j = 0
    while j < len(previous_tiles):
        tile = previous_tiles[j]
        if not intersects_with(tile[0], tile[1], tile[2], tile[3], local_tiles):
            local_tiles.append([tile[0], tile[1], tile[2], tile[3]])
            negative_count += 1
            save_tile(file_name, tile[0], tile[1], tile[2], tile[3], "{}/negative/{}-{}.jpg".format(output_path, i, negative_count))
            del previous_tiles[j]
        else:
            j += 1
