'''Simple two layer neural network (784-10).

Model contains 784 x 10 weights trained using batch gradient descent.
Output layer uses linear activation with argmax used only while testing.
'''

import numpy as np
from mnist_read import get_train_data, get_test_data

N_INPUTS, N_OUTPUTS = 784, 10
TRAIN_DATA_SIZE = 1_000

ALPHA = 0.001
BATCH_SIZE, NUM_ITER = 100, 500

def main() -> None:
    train_data = get_train_data(TRAIN_DATA_SIZE)
    test_data = get_test_data()
    weights = np.random.rand(N_OUTPUTS, N_INPUTS) * 0.002 - 0.001
    train_network(weights, train_data)
    test_network(weights, test_data)


def obtain_outputs(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    '''Obtain output layer from weights and input layer'''
    return weights.dot(x)

def update_weights(weights: np.ndarray, x: np.ndarray, y_exp: np.ndarray) -> None:
    '''Updates the weights using given batch of values.'''
    y_pred = np.tensordot(x, weights, axes=(1,1))
    y_delta = y_pred - y_exp
    w_delta = np.tensordot(y_delta, x, axes=(0,0))
    weights -= (ALPHA / BATCH_SIZE * w_delta)


def train_network(weights: np.ndarray, train_data: np.ndarray) -> None:
    '''Changes weights matrix using train_data.'''
    train_images, train_labels = train_data

    for iter_no in range(NUM_ITER):
        for batch_start in range(0, TRAIN_DATA_SIZE, BATCH_SIZE):
            image_batch = train_images[batch_start: batch_start + BATCH_SIZE]
            label_batch = train_labels[batch_start: batch_start + BATCH_SIZE]
            label_list = np.zeros((BATCH_SIZE, 10))
            label_list[np.arange(BATCH_SIZE), label_batch] = 1
            update_weights(weights, image_batch, label_list)

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
