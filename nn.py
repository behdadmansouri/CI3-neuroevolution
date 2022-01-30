import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.layer_sizes = layer_sizes
        self.w1 = np.random.normal(
            size=(layer_sizes[1], layer_sizes[0]))  # weights of layer 1 with normal initialization
        self.w2 = np.random.normal(
            size=(layer_sizes[2], layer_sizes[1]))  # weights of layer 2 with normal initialization
        self.b1 = np.zeros((layer_sizes[1], 1))
        self.b2 = np.zeros((layer_sizes[2], 1))

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        self.optimization(x)
        mid_layer = self.activation(self.w1 @ x + self.b1)
        output = self.activation(self.w2 @ mid_layer + self.b2)
        return output

    def optimization(self, x):
        if self.layer_sizes[0] >= 1:
            if x[0] > 400:
                x[0] = 1
            elif x[0] < 160:
                x[0] = 0
            else:
                x[0] = (x[0] - 177) / 410
        if self.layer_sizes[0] >= 2:
            x[1] /= 756
        if self.layer_sizes[0] >= 3:
            if x[0] > 400:
                x[2] = 1
            elif x[2] < 160:
                x[2] = 0
            else:
                x[2] = (x[2] - 177) / 410
        if self.layer_sizes[0] >= 4:
            x[3] /= 756
        if self.layer_sizes[0] >= 5:
            if x[4] > 400:
                x[4] = 1
            elif x[4] < 160:
                x[4] = 0
            else:
                x[4] = (x[4]-177)/410
        if self.layer_sizes[0] >= 6:
            x[5] /= 756
