'''Module containing functions to read the MNIST_data as numpy arrays.'''

import struct
import numpy as np

def load_mnist(images_path: str, labels_path: str) -> tuple[np.ndarray, np.ndarray]:
    '''Read the required data files and return corresponding numpy arrays.'''
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, n, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images/256, labels

def get_train_data(num_of_data: int) -> tuple[np.ndarray, np.ndarray]:
    '''Return required number of randomly selected training data.'''
    images_path = 'MNIST_data/train-images.idx3-ubyte'
    labels_path = 'MNIST_data/train-labels.idx1-ubyte'
    images, labels = load_mnist(images_path, labels_path)

    data_slice = np.random.permutation(len(images))[:num_of_data]
    return images[data_slice], labels[data_slice]

def get_test_data() -> tuple[np.ndarray, np.ndarray]:
    '''Return the testing dataset.'''
    images_path = 'MNIST_data/t10k-images.idx3-ubyte'
    labels_path = 'MNIST_data/t10k-labels.idx1-ubyte'
    images, labels = load_mnist(images_path, labels_path)

    return images, labels
