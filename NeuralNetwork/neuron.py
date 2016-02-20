import numpy as np

from NeuralNetwork.utilities import sigmoid

DEFAULT_BIAS = -1
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_THRESHOLD = 0


class Neuron:
    """Represents a neuron in a neural network.

    """

    def __init__(self, weights, bias=DEFAULT_BIAS, learning_rate=DEFAULT_LEARNING_RATE,
                 threshold=DEFAULT_THRESHOLD):
        """Initializes our neuron.

        :param weights: For future inputs, the weights associated. Index 0 is for the bias nodes weight.
        :param bias_weight:
        :param bias: Value of the bias.
        :param learning_rate:
        :param threshold
        """
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.threshold = threshold

    def activates(self, inputs):
        """Calculates whether or not the neron fires with given inputs.

        :param inputs: Array containing the inputs corresponding to the weights passed to the constructor
        :return: 1 if the neuron fires, 0 if it failed to fire
        """
        total = self.weights[0] * self.bias
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i + 1]
        return sigmoid(total)
