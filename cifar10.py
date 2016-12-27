import numpy as np
import tensorflow as tf
import prettytensor as pt
from cifar10_input import *
from cifar10_helper_functions import *

# Load data
path = "cifar-10/"
label_names = load_label_names(path)
x_data, data_labels_onehot, data_labels_int = load_batch(path, "data_batch_1")
x_test, test_labels_onehot, test_labels_int = load_batch(path, "test_batch")

# Initial variables
img_width = 32
img_height = 32
img_depth = 3
img_size = img_width * img_height * img_depth
img_shape = (img_width, img_height)
number_of_classes = 10

# Placeholders
x = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth]) # Input image
y_onehot = tf.placeholder(tf.float32, [None, number_of_classes])         # Input labels
y_int = tf.argmax(y_onehot, dimension=1)                                 # True labels

# Wrap input image in PrettyTensor wrapper
x_pretty = pt.wrap(x)

# Layers
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred_, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(class_count=number_of_classes, labels=y_onehot)

# Helper function to retrieve weights
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        weights = tf.get_variable('weights')
    return weights

# Weights
weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# Helper function to retrive layer output
def get_layer_output(layer_name):
    tensor_name = layer_name + "/Relu:0"
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
    return tensor

# Layer outputs
output_conv1 = get_layer_output(layer_name='layer_conv1')
output_conv2 = get_layer_output(layer_name='layer_conv2')

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

    batch_size = 64
    best_acc = 0.0
    iterations = 200000

    # Train network
    for i in range(iterations):
        # Get random batch
        rand_index = np.random.choice(len(x_data), batch_size, replace=False)
        x_batch = x_data[rand_index, :, :, :]
        y_batch = data_labels_onehot[rand_index, :]
        feed_dict_train = {x: x_batch, y_onehot: y_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            if acc > best_acc:
                best_acc = acc
                print("Iteration: {0}, Accuracy: {1:.2%} *".format(i, acc))
                saver.save(sess=session, save_path=save_path)
            else:
                print("Iteration: {0}, Accuracy: {1:.2%}".format(i, acc))

    # Test network accuracy
    test_batch_size = 256

    num_test = len(x_test)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)

        images = x_test[i:j, :]
        labels = test_labels_onehot[i:j, :]

        feed_dict = {x: images, y_onehot: labels}
        cls_pred[i:j] = session.run(y_pred, feed_dict=feed_dict)

        i = j

    cls_true = test_labels_int
    correct = (cls_true == cls_pred)
    correct_sum = np.count_nonzero(correct)

    acc = float(correct_sum) / num_test
    print("Accuracy on Test-Set: {0:.2%} ({1} / {2})".format(acc, correct_sum, num_test))

    # Plot 9 images from the data set
    # images = x_data[15:25]
    # labels = np.empty(10, dtype='object')
    # for i in range(15, 25): labels[i] = label_names[data_labels_int[i]]
    # labels = data_labels_int[15:25]
    # plot_sample(images, labels)
    # plot_images(images, labels)

    # Plot weights
    plot_conv_weights(session.run(weights_conv1), 0, "cifar_conv1_weights.png")
    plot_conv_weights(session.run(weights_conv2), 0, "cifar_conv2_weights.png")

    # Plot output
    feed_dict = {x: [x_test[16]]}
    values = session.run(output_conv1, feed_dict=feed_dict)
    plot_conv_layer(values, "cifar_conv1_outputs.png")
    values = session.run(output_conv2, feed_dict=feed_dict)
    plot_conv_layer(values, "cifar_conv2_outputs.png")

    # Confusion and miss classifications
    feed_dict = {x: x_data, y_onehot: data_labels_onehot, y_int: data_labels_int}
    correct, y_pred = session.run([correct_prediction, y_pred], feed_dict=feed_dict)

    # Plot confusion matrix
    print_confusion_matrix(data_labels_int, y_pred, number_of_classes)

    # Plot 9 misclassifications
    images = x_data[correct == False]
    y_true = data_labels_int[correct == False]
    y_pred = y_pred[correct == False]
    plot_images(images[0:5], y_true[0:5], y_pred[0:5])
