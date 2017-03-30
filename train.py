import yaml
import tensorflow as tf

config = yaml.safe_load(open("config.yml"))

width = config['training']['sample_width']
height = config['training']['sample_height']

train_input_path = config['training']['storage_path'] + "/train.tfrecords"
test_input_path = config['training']['storage_path'] + "/test.tfrecords"
log_path = config['training']['storage_path'] + "/train_log"
model_path = config['training']['storage_path'] + "/model"

def read_and_decode(filename_queue):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.float32)

    image_shape = tf.stack([height * width])
    image = tf.reshape(image, image_shape)
    label = tf.one_hot(features['label'], 2)


    images, labels = tf.train.shuffle_batch( [image, label],
                                             batch_size=100,
                                             capacity=500,
                                             num_threads=1,
                                             min_after_dequeue=1)

    return images, labels



train_queue = tf.train.string_input_producer([train_input_path], num_epochs=None)
train_image_batch, train_label_batch = read_and_decode(train_queue)

test_queue = tf.train.string_input_producer([test_input_path], num_epochs=None)
test_image_batch, test_label_batch = read_and_decode(test_queue)

x = tf.placeholder(tf.float32, [None, width*height])
y_ = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([width * height, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

tf.add_to_collection("y", y)


with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()


with tf.Session()  as sess:

    train_writer = tf.summary.FileWriter(log_path,sess.graph)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):
        batch_xs, batch_ys = sess.run([train_image_batch, train_label_batch])
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    batch_xs, batch_ys = sess.run([test_image_batch, test_label_batch])
    print("accuracy: {}".format(
        sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    ))

    saver = tf.train.Saver()
    saver.save(sess, model_path)
    saver.export_meta_graph(model_path + ".meta")

    coord.request_stop()
    coord.join(threads)
