import numpy as np
import tensorflow as tf
from MNIST_helper_functions import *

# Load MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# Print size of the data set
# print("Size of:")
# print("- Training-set:\t\t{}".format(len(data.train.labels)))
# print("- Test-set:\t\t{}".format(len(data.test.labels)))

# Convert from One-Hot to int
data.test.classif = np.array([label.argmax() for label in data.test.labels])
data.train.classif = np.array([label.argmax() for label in data.train.labels])
data.test.cls = np.argmax(data.test.labels, axis=1)

# Initial variables
img_width = 28
img_height = 28
img_size = img_width * img_height
img_shape = (img_width, img_height)
number_of_channels = 1
number_of_classes = 10

# Layer variables
conv1_filter_size = 5           # Convolutional layer 1 filter size
conv1_filters = 16              # Convolutional layer 1 number of filters

conv2_filter_size = 5           # Convolutional layer 2 filter size
conv2_filters = 36              # Convolutional layer 2 number of filters

fc_size = 128                   # Fully connected layer.

# Placeholders
x = tf.placeholder(tf.float32, [None, img_size])                 # Input image
y_onehot = tf.placeholder(tf.float32, [None, number_of_classes]) # Input labels

# Reshape image to 4-dim so it can be used in the layers
x_image = tf.reshape(x, [-1, img_width, img_height, number_of_channels])

# Layers
layer_conv1, weights_conv1 = \
    create_conv_layer(x_image, number_of_channels, conv1_filter_size,
                      conv1_filters, use_pooling=True)

layer_conv2, weights_conv2 = \
    create_conv_layer(layer_conv1, conv1_filters, conv2_filter_size,
                   conv2_filters, use_pooling=True)

layer_flat, number_of_features = create_flatten_layer(layer_conv2)

layer_fc1 = create_fc_layer(layer_flat, number_of_features,
                            fc_size, use_relu=True)

layer_fc2 = create_fc_layer(layer_fc1, fc_size, number_of_classes,
                            use_relu=False)

y_pred = tf.argmax(tf.nn.softmax(layer_fc2), dimension=1)   # Predicted class

# Optimization
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(layer_fc2, y_onehot)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Preformance messure
y_int = tf.argmax(y_onehot, dimension=1)                   # True labels
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

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            print("Iteration: {0}, Accuracy: {1:.2%}".format(i, acc))

    # Test network accuracy
    test_batch_size = 256

    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)

        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]

        feed_dict = {x: images, y_onehot: labels}
        cls_pred[i:j] = session.run(y_pred, feed_dict=feed_dict)

        i = j

    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = np.count_nonzero(correct)

    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.2%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot 9 images from the data set
    # images = data.test.images[30:40]
    # labels = data.test.classif[30:40]
    # plot_images(images, img_shape, labels)

    # Confusion and miss classifications
    feed_dict = {x: data.test.images, y_onehot: data.test.labels, y_int: data.test.classif}
    correct, y_pred = session.run([correct_prediction, y_pred], feed_dict=feed_dict)

    # Plot confusion matrix
    print_confusion_matrix(data.test.classif, y_pred, number_of_classes)

    # Plot 9 misclassifications
    images = data.test.images[correct == False]
    y_int = data.test.classif[correct == False]
    y_pred = y_pred[correct == False]

    plot_images(images[0:10], img_shape, y_int[0:10], y_pred[0:10])

    # Plot weights
    name = "conv1_weights_" + str(iterations) + ".png"
    plot_conv_weights(session.run(weights_conv1), 0, name)
    name = "conv2_weights_" + str(iterations) + ".png"
    plot_conv_weights(session.run(weights_conv2), 0, name)

    # Plot convolutional layers output
    feed_dict = {x: [data.test.images[30]]}
    values = session.run(layer_conv1, feed_dict=feed_dict)
    name = "conv1_outputs_" + str(iterations) + ".png"
    plot_conv_layer(values, name)
    values = session.run(layer_conv2, feed_dict=feed_dict)
    name = "conv2_outputs_" + str(iterations) + ".png"
    plot_conv_layer(values, name)
