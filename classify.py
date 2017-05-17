import tensorflow as tf
import model
import yaml
import glob
import os

config = yaml.safe_load(open("config.yml"))

input_path      = config['training']['storage_path'] + "/frames/*.png"
model_path      = config['training']['storage_path'] + "/model"
output_path     = config['training']['storage_path'] + "/classified"
width           = config['training']['sample_width']
height          = config['training']['sample_width']
num_channels    = 1 if config['training']['grayscale'] else 3
patch_width     = config['training']['glipse_width']
patch_height    = config['training']['glipse_height']

if not os.path.exists(output_path):
    os.makedirs(output_path)

x = model.x
y = model.y
y_ = model.y
keep_prob = model.keep_prob

saver = tf.train.import_meta_graph(model_path + ".meta")

file_names = glob.glob(input_path)
filename_queue = tf.train.string_input_producer(file_names)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

img = tf.image.decode_png(value)
float_img = tf.image.convert_image_dtype(img, tf.float32)

patches = tf.extract_image_patches(
    [float_img],
    [1, patch_width, patch_height, 1],
    [1, patch_width / 2, patch_height / 2, 1],
    [1, 1, 1, 1],
    "SAME"
)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver.restore(sess, tf.train.latest_checkpoint(config['training']['storage_path']))
    tf.get_collection('model')

    i = 0

    for j, example in enumerate(sess.run(patches)):
        for k, row in enumerate(example):
            for l, patch_data in enumerate(row):
                image_shape = tf.stack([patch_height, patch_width, 3])
                reshaped_img = tf.reshape(patch_data, image_shape)
                uint_img = tf.image.convert_image_dtype(reshaped_img, tf.uint8)
                png_data = tf.image.encode_png(uint_img)
                # sess.run(
                #     tf.write_file(output_path + "/{}.png".format(i), png_data)
                # )
                # print(patch_data)
                grayscale_img = tf.image.rgb_to_grayscale(reshaped_img)
                resized_img = tf.image.resize_images(
                    grayscale_img if config['training']['grayscale'] else reshaped_img,
                    [width, height]
                )
                image_shape = tf.stack([height * width * num_channels])
                image = tf.reshape(resized_img, image_shape)
                img = sess.run(image)
                print(sess.run(y, feed_dict={x: [img], keep_prob: 1.0}))
                i += 1

    coord.request_stop()
    coord.join(threads)
