import classifier
import sys
import yaml
from PIL import Image, ImageDraw, ImageColor
import colorsys

config     = yaml.safe_load(open("config.yml"))
w          = config['training']['patch_width']
h          = config['training']['patch_height']
input_path = sys.argv[1]

predictions = classifier.classify(input_path)

source = Image.open(input_path).convert('RGB')

heat_map = Image.new('RGB', source.size, (0,0,0))

draw = ImageDraw.Draw(heat_map, "RGBA")
for prediction, x, y in predictions:
    draw.rectangle(
        [x - w/2, y - h/2, x + w/2, y + h/2],
        fill=(255,255,255, int(prediction * 256))
    )

data = heat_map.load()
source_data = source.load()
for x in range(heat_map.size[0]):
    for y in range(heat_map.size[1]):
        h, l, s = colorsys.rgb_to_hls(
            source_data[x, y][0] / 255.0,
            source_data[x, y][1] / 255.0,
            source_data[x, y][2] / 255.0
        )
        r, g, b = colorsys.hls_to_rgb(
            data[x,y][0] * 0.4 / 256, l, 0.8
        )
        data[x, y] = (
            int(r * 255), int(g * 255), int(b * 255)
        )

clusters = classifier.cluster(predictions)

def draw_circle(draw, x, y, r):
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255,255,255))
    draw.ellipse((x - r - 0.5, y - r - 0.5, x + r + 0.5, y + r + 0.5), outline=(255,255,255))

for c in clusters:
    draw_circle(draw, c[0], c[1], c[2])

del draw

heat_map.show()
