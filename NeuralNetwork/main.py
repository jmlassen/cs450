import csv

import numpy as np

from sklearn import datasets

DEFAULT_BIAS = -1
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_THRESHOLD = 0
DEFAULT_WEIGHT_FLOOR = -1
DEFAULT_WEIGHT_CEILING = 1


class Neuron:
    """Represents a neuron in a neural network.

    """

    def __init__(self, weights, bias_weight, bias=DEFAULT_BIAS, learning_rate=DEFAULT_LEARNING_RATE,
                 threshold=DEFAULT_THRESHOLD):
        """Initializes our neuron.

        :param weights: For future inputs, the weights associated. Index 0 is for the bias nodes weight.
        :param bias_weight:
        :param bias: Value of the bias. Defaults to -1
        :param learning_rate:
        :param threshold
        """
        self.weights = weights
        self.bias_weight = bias_weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.threshold = threshold

    def fires(self, inputs):
        """Calculates whether or not the neron fires with given inputs.

        :param inputs: Array containing the inputs corresponding to the weights passed to the constructor
        :return: 1 if the neuron fires, 0 if it failed to fire
        """
        # Ensure size of inputs is the same size as weights.
        if len(inputs) != len(self.weights):
            raise Exception('Size of inputs ({}) must be consistent with size of weights ({})!'
                            .format(len(inputs), len(self.weights)))
        total = self.bias * self.bias_weight
        # Sum inputs
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]
        return 1 if total > self.threshold else 0


def get_starting_weights(n_inputs):
    return np.random.uniform(DEFAULT_WEIGHT_FLOOR, DEFAULT_WEIGHT_CEILING, n_inputs)


def run_iris():
    # Load iris data set
    iris = datasets.load_iris()
    n_inputs = len(iris.data[0])
    neuron = Neuron(get_starting_weights(n_inputs), get_starting_weights(1))
    for point in iris.data:
        fires = neuron.fires(point)
        print("{} = {}".format(point, fires))


def run_diabetes():
    data = []
    with open('pima-indians-diabetes.data') as diabetes_file:
        diabetes_reader = csv.reader(diabetes_file)
        for row in diabetes_reader:
            data.append([round(float(i), 2) for i in row[:8]])
    n_inputs = len(data[0])
    neuron = Neuron(get_starting_weights(n_inputs), get_starting_weights(1))
    for point in data:
        fires = neuron.fires(point)
        print("{} = {}".format(point, fires))


def main():
    run_iris()
    run_diabetes()


if __name__ == "__main__":
    main()
