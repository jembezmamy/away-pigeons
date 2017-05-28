import model
import tensorflow as tf
import numpy as np

log_path = model.log_path
model_path = model.model_path

x = model.x
y = model.y
y_ = model.y_
training = model.training
train_image_batch = model.train_image_batch
train_label_batch = model.train_label_batch
test_image_batch = model.test_image_batch
test_label_batch = model.test_label_batch

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

# train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
train_step = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")


with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    if model.output_count > 1:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    else:
        correct_prediction = tf.equal(tf.greater(y, 0.5), tf.greater(y_, 0.5))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

tf.summary.image("image", model.x_image)

merged = tf.summary.merge_all()


with tf.Session()  as sess:
    train_writer = tf.summary.FileWriter(log_path + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_path + '/test')

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver = tf.train.Saver()

    for i in range(5000):
        if i % 10 == 0:
            batch_xs, batch_ys = sess.run([test_image_batch, test_label_batch])
            summary, test_acc = sess.run([merged, accuracy], feed_dict={x: batch_xs, y_: batch_ys, training: False})
            test_writer.add_summary(summary, i)
            batch_xs, batch_ys, names = sess.run([train_image_batch, train_label_batch, model.train_name_batch])
            train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, training: False})
            print('Accuracy at step %s: %s / %s' % (i, train_acc, test_acc))

            predictions = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, training: False})
            # predictions = np.hstack([predictions, np.transpose(np.matrix(names))])
            # print(predictions)
        batch_xs, batch_ys = sess.run([train_image_batch, train_label_batch])
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys, training: True})
        train_writer.add_summary(summary, i)
        if i % 100 == 0:
            saver.save(sess, model_path)

    coord.request_stop()
    coord.join(threads)
