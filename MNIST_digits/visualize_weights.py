'''Visualization of how each pixel affects the sum for each digit.

Uses the simple neural network from 'two_layer.py'.
Model contains 784 x 10 weights trained using batch gradient descent.
The 784 weights assosciated with each digit is reshaped into a 28x28 image.
'''

import numpy as np
import matplotlib.pyplot as plt
from mnist_read import get_train_data

N_INPUTS, N_OUTPUTS = 784, 10
TRAIN_DATA_SIZE = 1_000

ALPHA = 0.0001
BATCH_SIZE, NUM_ITER = 10, 200

def main() -> None:
    train_data = get_train_data(TRAIN_DATA_SIZE)
    weights = np.zeros((N_OUTPUTS, N_INPUTS))
    train_network(weights, train_data)
    display_weights(weights)

def obtain_outputs(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    '''Obtain output layer from weights and input layer'''
    return weights.dot(x)

def update_weights(weights: np.ndarray, x: np.ndarray, y_exp: np.ndarray) -> np.ndarray:
    '''Return the desired weight delta for given values.'''
    y_pred = weights.dot(x)
    y_delta = (y_pred - y_exp).reshape(N_OUTPUTS, 1)
    x = x.reshape(1, N_INPUTS)
    return y_delta.dot(x)


def train_network(weights: np.ndarray, train_data: np.ndarray) -> None:
    '''Changes weights matrix using train_data.'''
    train_images, train_labels = train_data

    for iter_no in range(NUM_ITER):
        weight_deltas = np.zeros((N_OUTPUTS, N_INPUTS))
        no_data = 0
        for image, label in zip(train_images, train_labels):
            label_list = np.zeros(10)
            label_list[label] = 1
            weight_deltas += update_weights(weights, image, label_list)
            no_data += 1
            if no_data % BATCH_SIZE == 0:
                weights -= (ALPHA / BATCH_SIZE * weight_deltas)
                weight_deltas = np.zeros((N_OUTPUTS, N_INPUTS))

        print(f'iterations: {iter_no+1}', end ='\r')
    print()


def display_weights(weights: np.ndarray) -> None:
    '''Displays the weight associated with each pixel for every digit.'''
    plot = plt.figure(figsize=(7, 6))
    for digit, weight in enumerate(weights, start=1):
        plot.add_subplot(3, 4, digit)
        plt.imshow(weight.reshape(28, 28))
        plt.axis('off')
        plt.title(f"{digit}")
    plt.show()


if __name__ == '__main__':
    main()
