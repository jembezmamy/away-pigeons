import yaml
import tensorflow as tf

config = yaml.safe_load(open("config.yml"))

width = config['training']['sample_width']
height = config['training']['sample_height']

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


# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)
#
# def bias_variable(shape):
#   initial = tf.constant(0.1, shape=shape)
#   return tf.Variable(initial)
#
# def conv2d(x, W):
#   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')



train_queue = tf.train.string_input_producer([train_input_path])
train_image_batch, train_label_batch, train_name_batch = read_and_decode(train_queue, True)

test_queue = tf.train.string_input_producer([test_input_path])
test_image_batch, test_label_batch, test_name_batch = read_and_decode(test_queue)



x = tf.placeholder(tf.float32, [None, width*height*num_channels])
y_ = tf.placeholder(tf.float32, [None, output_count])

x_image = tf.reshape(x, [-1,width,height,num_channels])

# W_conv1 = weight_variable([5, 5, num_channels, 32])
# b_conv1 = bias_variable([32])
# x_image = tf.reshape(x, [-1,width,height,num_channels])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# # W_conv2 = weight_variable([5, 5, 32, 64])
# # b_conv2 = bias_variable([64])
# # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# # h_pool2 = max_pool_2x2(h_conv2)
#
# W_fc1 = weight_variable([width * height * 64 / 16, 1024])
# b_fc1 = bias_variable([1024])
# h_pool2_flat = tf.reshape(h_pool1, [-1, width * height * 64 / 16])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W = weight_variable([width * height, output_count])
# b = bias_variable([output_count])
# y = tf.matmul(h_fc1_drop, W) + b

# W = weight_variable([width * height * num_channels, output_count])
# b = weight_variable([output_count])
# y = tf.matmul(x, W) + b

training = tf.placeholder(tf.bool)

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
y = tf.layers.dense(inputs=dropout, units=output_count)
