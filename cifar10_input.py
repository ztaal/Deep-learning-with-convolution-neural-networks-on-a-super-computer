import pickle
import numpy as np
import tensorflow as tf
import os

def load_label_names(path):

    # Load data
    with open(path + "batches.meta", mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Load label names
    label_names = data[b'label_names']

    # Convert from binary strings
    label_names = [i.decode('utf-8') for i in label_names]

    return label_names

def load_batch(path, filename):
    img_width = 32
    img_height = 32
    img_depth = 3
    number_of_classes = 10

    # Load data
    with open(path + filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Images
    images = data[b'data'].reshape([-1, img_depth, img_width, img_height])
    images = images.transpose([0, 2, 3, 1])

    # Classifiers
    classif = np.array(data[b'labels'])

    # Onehot classifiers
    classif_onehot = np.eye(number_of_classes)[classif]

    return images, classif_onehot, classif
