'''Three layer neural network (784-N-10) with relu in hidden layer.

Model contains 784*N + N*10 weights trained using batch gradient descent.
Hidden layer uses relu as the non linear activation function.
Output layer uses linear activation with argmax used only while testing.
'''

import numpy as np
from mnist_read import get_train_data, get_test_data

N_INPUTS, N_OUTPUTS = 784, 10
N_HIDDEN = 16
TRAIN_DATA_SIZE = 1_000

ALPHA = 0.002
BATCH_SIZE, NUM_ITER = 10, 500

def main() -> None:
    train_data = get_train_data(TRAIN_DATA_SIZE)
    test_data = get_test_data()
    weights_0_1 = np.random.rand(N_HIDDEN, N_INPUTS) * 0.002 - 0.001
    weights_1_2 = np.random.rand(N_OUTPUTS, N_HIDDEN) * 0.002 - 0.001
    train_network(weights_0_1, weights_1_2, train_data)
    test_network(weights_0_1, weights_1_2, test_data)


def relu(x: np.ndarray) -> np.ndarray:
    return (x>0) * x

def relu_der(x: np.ndarray) -> np.ndarray:
    return x>0

def obtain_outputs(weights_0_1: np.ndarray, weights_1_2: np.ndarray, layer0: np.ndarray) -> np.ndarray:
    '''Obtain output layer from weights and input layer'''
    layer1 = relu(weights_0_1.dot(layer0))
    layer2 = weights_1_2.dot(layer1)
    return layer2

def update_weights(weights_0_1: np.ndarray, weights_1_2: np.ndarray, 
                   layer0: np.ndarray, layer2_exp: np.ndarray) -> np.ndarray:
    '''Return the desired weight delta for given values.'''
    layer1 = relu(weights_0_1.dot(layer0))
    layer2 = weights_1_2.dot(layer1)
    delta2 = layer2 - layer2_exp
    delta1 = weights_1_2.T.dot(delta2.reshape(N_OUTPUTS, 1)).T  * relu_der(layer1)
    weights_1_2_delta = delta2.reshape(N_OUTPUTS, 1).dot(layer1.reshape(1, N_HIDDEN))
    weights_0_1_delta = delta1.reshape(N_HIDDEN, 1).dot(layer0.reshape(1, N_INPUTS))
    return weights_0_1_delta, weights_1_2_delta


def train_network(weights_0_1: np.ndarray, weights_1_2: np.ndarray, train_data: np.ndarray) -> None:
    '''Changes weights matrix using train_data.'''
    train_images, train_labels = train_data

    for iter_no in range(NUM_ITER):
        weight_deltas_0_1 = np.zeros((N_HIDDEN, N_INPUTS))
        weight_deltas_1_2 = np.zeros((N_OUTPUTS, N_HIDDEN))
        no_data = 0
        for image, label in zip(train_images, train_labels):
            label_list = np.zeros(10)
            label_list[label] = 1
            weights_deltas = update_weights(weights_0_1, weights_1_2, image, label_list)
            weight_deltas_0_1 += weights_deltas[0]
            weight_deltas_1_2 += weights_deltas[1]
            no_data += 1
            if no_data % BATCH_SIZE == 0:
                weights_0_1 -= (ALPHA / BATCH_SIZE * weight_deltas_0_1)
                weights_1_2 -= (ALPHA / BATCH_SIZE * weight_deltas_1_2)
                weight_deltas_0_1 = np.zeros((N_HIDDEN, N_INPUTS))
                weight_deltas_1_2 = np.zeros((N_OUTPUTS, N_HIDDEN))

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
