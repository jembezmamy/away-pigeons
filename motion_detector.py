from PIL import Image
from PIL import ImageChops

def detect(image_1, image_2, threshold = 64, min_area = 0.01):
    return get_amount_of_motion(image_1, image_2, threshold, min_area) >= 1

def get_amount_of_motion(image_1, image_2, threshold = 64, min_area = 0.01):
    histogram = ImageChops.difference(image_1, image_2).histogram()
    sum_r = sum(histogram[threshold:256])
    sum_g = sum(histogram[(threshold + 256):(256 * 2)])
    sum_b = sum(histogram[(threshold + 256 * 2):(256 * 3)])
    changed_area = (sum_r + sum_g + sum_b) / 3
    return changed_area / (min_area * image_1.size[0] * image_1.size[1])
