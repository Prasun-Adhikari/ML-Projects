'''Three layer neural network (784-N-10) with tanh, softmax and dropout.

Model contains 784*N + N*10 weights trained using batch gradient descent.
Hidden layer uses tanh along with a 50% dropout.
Output layer uses softmax with base e.
'''

import numpy as np
from mnist_read import get_train_data, get_test_data

N_INPUTS, N_OUTPUTS = 784, 10
N_HIDDEN = 16
TRAIN_DATA_SIZE = 1_000

ALPHA = 0.1
BATCH_SIZE, NUM_ITER = 100, 500

def main() -> None:
    train_data = get_train_data(TRAIN_DATA_SIZE)
    test_data = get_test_data()
    weights_0_1 = np.random.rand(N_HIDDEN, N_INPUTS) * 0.2 - 0.1
    weights_1_2 = np.random.rand(N_OUTPUTS, N_HIDDEN) * 0.2 - 0.1
    train_network(weights_0_1, weights_1_2, train_data)
    test_network(weights_0_1, weights_1_2, test_data)


def tanh(x): 
    return np.tanh(x)

def tanh_der(y):
    return 1 - y**2

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1).reshape(-1, 1)

def obtain_outputs(weights_0_1: np.ndarray, weights_1_2: np.ndarray, layer0: np.ndarray) -> np.ndarray:
    '''Obtain output layer from weights and input layer'''
    layer1 = tanh(weights_0_1.dot(layer0))
    layer2 = weights_1_2.dot(layer1)
    return layer2

def update_weights(weights_0_1: np.ndarray, weights_1_2: np.ndarray, 
                   layer0: np.ndarray, layer2_exp: np.ndarray) -> None:
    '''Updates the weights using given batch of values.'''
    
    dropout_mask = np.random.randint(2, size=N_HIDDEN)
    layer1 = tanh(np.tensordot(layer0, weights_0_1, axes=(1,1))) * dropout_mask * 2
    layer2 = softmax(np.tensordot(layer1, weights_1_2, axes=(1,1)))
    delta2 = (layer2 - layer2_exp) / layer2.shape[1]
    delta1 = np.tensordot(delta2, weights_1_2, axes=(1,0)) * tanh_der(layer1) * dropout_mask
    
    weights_1_2_delta = np.tensordot(delta2, layer1, axes=(0,0))
    weights_0_1_delta = np.tensordot(delta1, layer0, axes=(0,0))
    weights_1_2 -= (ALPHA / BATCH_SIZE * weights_1_2_delta)
    weights_0_1 -= (ALPHA / BATCH_SIZE * weights_0_1_delta)


def train_network(weights_0_1: np.ndarray, weights_1_2: np.ndarray, train_data: np.ndarray) -> None:
    '''Changes weights matrix using train_data.'''
    train_images, train_labels = train_data

    for iter_no in range(NUM_ITER):
        for batch_start in range(0, TRAIN_DATA_SIZE, BATCH_SIZE):
            image_batch = train_images[batch_start: batch_start + BATCH_SIZE]
            label_batch = train_labels[batch_start: batch_start + BATCH_SIZE]
            label_list = np.zeros((BATCH_SIZE, 10))
            label_list[np.arange(BATCH_SIZE), label_batch] = 1
            update_weights(weights_0_1, weights_1_2, image_batch, label_list)

        print(f'iterations: {iter_no+1}', end ='\r')
    print()


def test_network(weights_0_1: np.ndarray, weights_1_2: np.ndarray, test_data: np.ndarray) -> None:
    '''Tests and displays the accuracy of the network using test_data.'''
    test_images, test_labels = test_data

    total_no = len(test_images)
    correct_no = 0
    for image, label in zip(test_images, test_labels):
        y_pred = obtain_outputs(weights_0_1, weights_1_2, image)
        if label == np.argmax(y_pred):
            correct_no += 1

    print(f'{correct_no:,} / {total_no:,} correct')
    print(f"accuracy: {correct_no/total_no*100:.4} %")


if __name__ == '__main__':
    main()
