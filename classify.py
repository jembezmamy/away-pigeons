import tensorflow as tf
import model
import yaml
import sys
import numpy as np
from PIL import Image

config = yaml.safe_load(open("config.yml"))

input_path      = sys.argv[1]
model_path      = config['training']['storage_path'] + "/model"
output_path     = config['training']['storage_path'] + "/classified"
width           = config['training']['example_width']
height          = config['training']['example_width']
num_channels    = 1 if config['training']['grayscale'] else 3
patch_width     = config['training']['patch_width']
patch_height    = config['training']['patch_height']

file_name = tf.placeholder(tf.string)
img_file = tf.read_file(file_name)
img = tf.image.decode_png(img_file)
if config['training']['grayscale']:
    img = tf.image.rgb_to_grayscale(img)
patches = tf.extract_image_patches([img],
     ksizes=[1, patch_height, patch_width, 1],
     strides=[1, patch_height/4, patch_width/4, 1],
     rates=[1,1,1,1],
     padding="VALID")
patches_shape = tf.shape(patches)
patches = tf.reshape(patches, [-1, patch_height, patch_width, num_channels])
patches = tf.image.resize_images(patches, [height, width])
patches = tf.reshape(patches, [-1, height * width * num_channels])

x = model.x
y = model.y
training = model.training

y_predictions = tf.nn.softmax(y)

saver = tf.train.Saver()

with tf.Session()  as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver.restore(sess, tf.train.latest_checkpoint(config['training']['storage_path']))

    batch_xs, shape = sess.run([patches, patches_shape], feed_dict={file_name: input_path})
    predictions = sess.run(y_predictions, feed_dict={x: batch_xs, training: False})
    v_count = shape[1]

    predictions = [[
        p[1],
        int((i % v_count + 0.5) * patch_width),
        int((i / v_count + 0.5) * patch_height)
    ] for i, p in enumerate(predictions)]
    predictions.sort(key=lambda tup: tup[0], reverse=True)
    print predictions

    coord.request_stop()
    coord.join(threads)
