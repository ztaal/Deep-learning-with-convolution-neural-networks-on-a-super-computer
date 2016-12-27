import os
import numpy as np
import tensorflow as tf
import prettytensor as pt
from MNIST_helper_functions import *

# Load MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# Convert from One-Hot to int
data.test.classif = np.argmax(data.test.labels, axis=1)
data.validation.classif = np.argmax(data.validation.labels, axis=1)

# Initial variables
img_width = 28
img_height = 28
img_size = img_width * img_height
img_shape = (img_width, img_height)
number_of_channels = 1
number_of_classes = 10

# Placeholders
x = tf.placeholder(tf.float32, [None, img_size])                 # Input image
y_onehot = tf.placeholder(tf.float32, [None, number_of_classes]) # Input labels
y_int = tf.argmax(y_onehot, dimension=1)                         # True labels

# Reshape image to 4-dim so it can be used in the layers
x_image = tf.reshape(x, [-1, img_width, img_height, number_of_channels])

# Wrap input image in PrettyTensor wrapper
x_pretty = pt.wrap(x_image)

# Layers
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred_, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(class_count=10, labels=y_onehot)

# Helper function to retrieve weights
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        weights = tf.get_variable('weights')
    return weights

# Weights
weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Preformance messure
y_pred = tf.argmax(y_pred_, dimension=1)              # Predicted labels
correct_prediction = tf.equal(y_pred, y_int)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saving
saver = tf.train.Saver()
save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = save_dir + 'weights'

# Session
with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    # Restore session with all variables
    saver.restore(sess=session, save_path=save_path)

    batch_size = 64

    # Test network (run one more iteration)
    # x_batch, y_batch = data.validation.next_batch(batch_size)
    # feed_dict_train = {x: x_batch, y_onehot: y_batch}
    # acc = session.run(accuracy, feed_dict=feed_dict_train)
    # print("Accuracy: {0:.2%}".format(acc))

    # Apply CNN on 9 images with the restored weights
    images = data.test.images[0:9]
    feed_dict_test = {x: images}
    true_labels_test = session.run(y_pred, feed_dict=feed_dict_test)
    print(true_labels_test)

    # Plot 9 images from the data set
    # images = data.test.images[0:9]
    # labels = data.test.classif[0:9]
    # plot_images(images, img_shape, labels)

    # Plot 9 misclassifications
    # feed_dict = {x: data.test.images, y_onehot: data.test.labels,
    #                   y_int: data.test.classif}
    # correct, y_pred = session.run([correct_prediction, y_pred],
    #                               feed_dict=feed_dict)
    #
    # images = data.test.images[correct == False]
    # y_int = data.test.classif[correct == False]
    # y_pred = y_pred[correct == False]
    #
    # plot_images(images[0:9], img_shape, y_int[0:9], y_pred[0:9])

    # Plot weights
    plot_conv_weights(session.run(weights_conv1))
    plot_conv_weights(session.run(weights_conv2))
