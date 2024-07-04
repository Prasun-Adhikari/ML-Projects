# MNIST digit recognition

## About MNIST
The MNIST database consists of a large collection of handwritten digits along with labels.
The digits are grayscale and normalized to fit in a 24x24 grid. The full database contains
60,000 training data and 10,000 test data.

Visit [The MNIST webpage](http://yann.lecun.com/exdb/mnist/) for more information and the
actual data files.


## Goal
The general goal is to classify the handwritten image into the correct digit.
My goal for this project was to create and train neural networks of various types and complexities from scratch.
I wanted to use 1,000 images to train the network and calculate accuracy using all 10,000 test images.


## Networks created
- 2-layer (784-10) linear
- 3-layer (784-N-10) relu
- 3-layer (784-N-10) relu + dropout


## References
I created these networks when studying the book ['grokking Deep Learning'](https://github.com/iamtrask/Grokking-Deep-Learning) by Andrew W. Trask.
Although the relevant theory and approaches are inspired from the book, the programs
were written by myself.

The code to convert the dataset (stored as .idx1-ubyte and .idx3-ubyte) to numpy arrays came from [this article](https://notebook.community/rasbt/python-machine-learning-book/code/bonus/reading_mnist).