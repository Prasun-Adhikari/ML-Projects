'''Simple two layer neural network (784-10).

Model contains 784 x 10 weights trained using batch gradient descent.
Output layer uses linear activation with argmax used only while testing.
'''

import numpy as np
from mnist_read import get_train_data, get_test_data

N_INPUTS, N_OUTPUTS = 784, 10
TRAIN_DATA_SIZE = 1_000

ALPHA = 0.0001
BATCH_SIZE, NUM_ITER = 10, 500

def main() -> None:
    train_data = get_train_data(TRAIN_DATA_SIZE)
    test_data = get_test_data()
    weights = np.random.rand(N_OUTPUTS, N_INPUTS) * 0.002 - 0.001
    train_network(weights, train_data)
    test_network(weights, test_data)


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


def test_network(weights: np.ndarray, test_data: np.ndarray) -> None:
    '''Tests and displays the accuracy of the network using test_data.'''
    test_images, test_labels = test_data

    total_no = len(test_images)
    correct_no = 0
    for image, label in zip(test_images, test_labels):
        y_pred = obtain_outputs(weights, image)
        if label == np.argmax(y_pred):
            correct_no += 1

    print(f'{correct_no:,} / {total_no:,} correct')
    print(f"accuracy: {correct_no/total_no*100:.4} %")


if __name__ == '__main__':
    main()
