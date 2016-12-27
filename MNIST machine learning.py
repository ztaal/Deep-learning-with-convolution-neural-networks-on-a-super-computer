import numpy as np
import tensorflow as tf
from MNIST_helper_functions import plot_images
from MNIST_helper_functions import plot_weight
from MNIST_helper_functions import plot_weights
from MNIST_helper_functions import print_confusion_matrix

# Load MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# Print size of the data set
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))

# Get labels as integers
data.test.classif = np.array([label.argmax() for label in data.test.labels])

# Initial variables
img_width = 28
img_height = 28
img_size = img_width * img_height
img_shape = (img_width, img_height)
number_of_classes = 10

# Placeholders
x = tf.placeholder(tf.float32, [None, img_size])                 # Input image
y_onehot = tf.placeholder(tf.float32, [None, number_of_classes]) # Input labels
y_int = tf.placeholder(tf.int64, [None])                   # Input labels as int

# Model variables
weights = tf.Variable(tf.zeros([img_size, number_of_classes]))
biases = tf.Variable(tf.zeros([number_of_classes]))

# Linear model
logits = tf.matmul(x, weights) + biases

# Optimization
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_onehot)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# Preformance messure
y_pred = tf.argmax(tf.nn.softmax(logits), dimension=1)      # Predicted classif
correct_prediction = tf.equal(y_pred, y_int)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Session
with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    batch_size = 64
    iterations = 10000
    # Train network
    for i in range(iterations):
        x_batch, y_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_onehot: y_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 1000 == 0:
            print("Iteration: {0}".format(i))

    # Accuracy
    feed_dict_test = {x: data.test.images, y_onehot: data.test.labels,
    y_int: data.test.classif}
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Classification Accuracy: {0:.2%}".format(acc))

    # Plot 10 images from the data set
    # images = data.test.images[0:10]
    # y_int = data.test.classif[0:10]
    # plot_images(images, img_shape, y_int)

    # Plot 10 misclassifications
    # correct, y_pred = session.run([correct_prediction, y_pred],
    #                               feed_dict=feed_dict_test)
    #
    # images = data.test.images[correct == False]
    # y_int = data.test.classif[correct == False]
    # y_pred = y_pred[correct == False]
    #
    # plot_images(images[0:10], img_shape, y_int[0:10], y_pred[0:10])

    # Plot weights
    # weights = session.run(weights)
    # plot_weights(weights, img_shape)
    #
    # weights = session.run(weights)
    # plot_weight(weights, img_shape, 3, iterations)

    # Plot confusion matrix
    # y_pred = session.run(y_pred, feed_dict=feed_dict_test)
    # print_confusion_matrix(data.test.classif, y_pred, number_of_classes)
