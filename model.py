import yaml
import tensorflow as tf

config = yaml.safe_load(open("config.yml"))

width = config['training']['example_width']
height = config['training']['example_height']

train_input_path = config['training']['storage_path'] + "/train.tfrecords"
test_input_path = config['training']['storage_path'] + "/test.tfrecords"
log_path = config['training']['storage_path'] + "/log"
model_path = config['training']['storage_path'] + "/model"
num_channels = 1 if config['training']['grayscale'] else 3
output_count = 2

def read_and_decode(filename_queue, distort = False):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'name': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    img_height = tf.cast(features['height'], tf.int32)
    img_width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, tf.stack([img_height, img_width, num_channels]))


    if distort:
        image = tf.random_crop(image, [height, width, num_channels])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, width, height)

    image = tf.image.per_image_standardization(image)

    image_shape = tf.stack([height * width * num_channels])
    image = tf.reshape(image, image_shape)
    label = tf.one_hot(features['label'], output_count)

    name = features['name']


    images, labels, names = tf.train.shuffle_batch( [image, label, name],
                                             batch_size=50,
                                             capacity=500,
                                             num_threads=3,
                                             min_after_dequeue=1)

    return images, labels, names



train_queue = tf.train.string_input_producer([train_input_path])
train_image_batch, train_label_batch, train_name_batch = read_and_decode(train_queue, True)

test_queue = tf.train.string_input_producer([test_input_path])
test_image_batch, test_label_batch, test_name_batch = read_and_decode(test_queue)



x = tf.placeholder(tf.float32, [None, width*height*num_channels], name="x")
y_ = tf.placeholder(tf.float32, [None, output_count])

x_image = tf.reshape(x, [-1,width,height,num_channels])

training = tf.placeholder(tf.bool, name="training")

# Convolutional Layer #1
conv1 = tf.layers.conv2d(
  inputs=x_image,
  filters=32,
  kernel_size=[5, 5],
  padding="same",
  activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
  inputs=pool1,
  filters=64,
  kernel_size=[5, 5],
  padding="same",
  activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2_flat = tf.reshape(pool2, [-1, (width / 4) * (height / 4) * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(
  inputs=dense, rate=0.4, training=training)

# Logits Layer
y = tf.layers.dense(inputs=dropout, units=output_count, name="y")
