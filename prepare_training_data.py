import yaml
import glob
import subprocess
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import math

config = yaml.safe_load(open("config.yml"))

input_path = config['training']['storage_path']
tmp_file_name = config['training']['storage_path'] + "/tmp.png"
train_output_path = config['training']['storage_path'] + "/train.tfrecords"
test_output_path = config['training']['storage_path'] + "/test.tfrecords"

width = config['training']['sample_width']
height = config['training']['sample_height']

positive_file_names = glob.glob(input_path + "/positive/*.png")
negative_file_names = glob.glob(input_path + "/negative/*.png")
random.shuffle(positive_file_names)
random.shuffle(negative_file_names)

positive_threshold = math.ceil(len(positive_file_names) * 0.8)
negative_threshold = math.ceil(len(negative_file_names) * 0.8)

file_names = positive_file_names + negative_file_names
labels = len(positive_file_names) * [1] + len(negative_file_names) * [0]

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


filename_queue = tf.train.string_input_producer(file_names)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value)
rgb_img = tf.image.convert_image_dtype(my_img, tf.float32)
grayscale_img = tf.image.rgb_to_grayscale(rgb_img)
resized_img = tf.image.resize_images(grayscale_img, [width, height])

train_writer = tf.python_io.TFRecordWriter(train_output_path)
test_writer = tf.python_io.TFRecordWriter(test_output_path)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    positive_example_count = 0
    negative_example_count = 0

    for i in range(len(file_names)):
        image = resized_img.eval()
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(width),
            'width': _int64_feature(height),
            'label': _int64_feature(labels[i]),
            'image_raw': _bytes_feature(image_raw)}))
        if labels[i] == 1:
            positive_example_count += 1
            writer = train_writer if positive_example_count < positive_threshold else test_writer
        else:
            negative_example_count += 1
            writer = train_writer if negative_example_count < negative_threshold else test_writer
        writer.write(example.SerializeToString())

    train_writer.close()
    test_writer.close()
    filename_queue.close()
    coord.request_stop()
    coord.join(threads)
