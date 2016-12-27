import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Returns a list of shape with random weights
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# Returns a list of length with random biases
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# Create a convolutional layer
def create_conv_layer(input, number_of_input_channels, filter_size,
                      number_of_filters, use_pooling=True):

    # Shape of the filter weights
    shape = [filter_size, filter_size, number_of_input_channels,
             number_of_filters]

    # Get random values for the weight and biases
    weights = new_weights(shape=shape)
    biases = new_biases(length=number_of_filters)

    # Create convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add bias to the result of the convolutional layer e.g y = w*x + b
    layer += biases

    # Down sample image if pooling is used
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')

    # Apply relu to remove negative values
    layer = tf.nn.relu(layer)

    # Return the layer and weights
    return layer, weights

# Create a flatten layer
def create_flatten_layer(layer):

    # Get the shape of the input layer
    layer_shape = layer.get_shape()

    # The number of features = img_height * img_width * number_of_channels
    number_of_features = layer_shape[1:4].num_elements()

    # Flatten layer
    layer_flat = tf.reshape(layer, [-1, number_of_features])

    # Return flattened layer and the number of features.
    return layer_flat, number_of_features

# Create a fully connected layer
def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):

    # Get random values for the weight and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate linear model
    layer = tf.matmul(input, weights) + biases

    # Apply relu
    if use_relu:
        layer = tf.nn.relu(layer)

    # Return layer
    return layer

# Plots 10 images in a 3x3 subplot with their true labels
def plot_images(images, img_shape, true_classif, pred_classif=None):
    assert len(images) == len(true_classif) == 10

    # Create subplots
    fig, axes = plt.subplots(1, 5)
    plt.rcParams.update({'font.size': 20})
    # Run through images
    for i, plot in enumerate(axes.flat):
        # Plot images
        plot.imshow(images[i].reshape(img_shape), cmap='binary')

        # Add label
        if pred_classif == None:
            xlabel = "True: {0}".format(true_classif[i])
        else:
            xlabel = "True: {0}\nPred: {1}".format(true_classif[i],
                                                 pred_classif[i])
        # Add label
        plot.set_xlabel(xlabel)

        # Remove ticks from plot
        plot.set_xticks([])
        plot.set_yticks([])

    # Show plot
    plt.subplots_adjust(wspace=0.1, hspace=-0.60)
    fig.tight_layout()
    # fig.savefig('MNIST_conv_miss_classifications.png', bbox_inches='tight')
    plt.show()

# Plot the weights
def plot_weights(weights, img_shape):

    # Get min and max values
    min_weight = np.min(weights)
    max_weight = np.max(weights)

    # Create subplots
    fig, axes = plt.subplots(2, 5)

    # Plot image.
    for i, plot in enumerate(axes.flat):
        if i<10:
            image = weights[:, i].reshape(img_shape)

            # Plot image.
            plot.imshow(image, vmin=min_weight, vmax=max_weight, cmap='seismic')

            # Add label
            plot.set_xlabel("Class: {0}".format(i))

        # Remove ticks from plot
        plot.set_xticks([])
        plot.set_yticks([])

    # Show plot
    fig.tight_layout()
    plt.show()
    # fig.savefig('Weights.png', bbox_inches='tight')

# Plot the a single weight
def plot_weight(weights, img_shape, index=0, iteration=0):

    # Get min and max values
    min_weight = np.min(weights)
    max_weight = np.max(weights)

    # Plot image.
    image = weights[:, index].reshape(img_shape)
    plt.imshow(image, vmin=min_weight, vmax=max_weight, cmap='seismic')
    plt.axis('off')

    # Show plot
    # name = "3_" + str(iteration) + ".png"
    # plt.savefig(name, bbox_inches='tight')
    plt.show()

# Plot weights for convolutional layers
def plot_conv_weights(weights, input_channel=0, name='na'):

    # Get min and max values
    min_weight = np.min(weights)
    max_weight = np.max(weights)

    number_of_filters = weights.shape[3]
    number_of_grids = math.ceil(math.sqrt(number_of_filters))

    # Create subplots
    fig, axis = plt.subplots(number_of_grids, number_of_grids)

    for i, plot in enumerate(axis.flat):
        if i < number_of_filters:

            # Get weights
            image = weights[:, :, input_channel, i]

            # Plot image.
            plot.imshow(image, vmin=min_weight, vmax=max_weight,
                        interpolation='nearest', cmap='seismic')

        # Remove ticks from plot
        plot.set_xticks([])
        plot.set_yticks([])

    # Show plot
    # fig.savefig(name, bbox_inches='tight')
    plt.show()

# Plots the output of the convolutional layer
def plot_conv_layer(layer, name):

    number_of_filters = layer.shape[3]
    number_of_grids = math.ceil(math.sqrt(number_of_filters))

    # Create subplots
    fig, axes = plt.subplots(number_of_grids, number_of_grids)

    for i, plot in enumerate(axes.flat):
        if i < number_of_filters:
            image = layer[0, :, :, i]

            # Plot image
            plot.imshow(image, interpolation='nearest', cmap='binary')

        # Remove ticks from plot
        plot.set_xticks([])
        plot.set_yticks([])

    # Show plot
    # fig.savefig(name, bbox_inches='tight')
    plt.show()

def print_confusion_matrix(true_classif, pred_classif, number_of_classes):

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=true_classif,
                          y_pred=pred_classif)

    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(number_of_classes)
    plt.xticks(tick_marks, range(number_of_classes))
    plt.yticks(tick_marks, range(number_of_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Show plot
    # plt.savefig('conv_confusion_matrix.png', bbox_inches='tight')
    plt.show()
