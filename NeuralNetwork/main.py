import csv
from sklearn import datasets

from NeuralNetwork.neural_network import NeuralNetwork
from NeuralNetwork.neuron import Neuron
from NeuralNetwork.utilities import get_starting_weights, cross_val_score


def run_iris():
    # Load iris data set
    iris = datasets.load_iris()
    n_inputs = len(iris.data[0])
    n_outputs = len(iris.target_names)
    network = NeuralNetwork()
    network.create_network(n_inputs, n_outputs, (3, 4))
    print("Neural network results accuracy: {}".format(cross_val_score(network, iris.data, iris.target, 3)))


def run_diabetes():
    data = []
    with open('pima-indians-diabetes.data') as diabetes_file:
        diabetes_reader = csv.reader(diabetes_file)
        for row in diabetes_reader:
            data.append([round(float(i), 2) for i in row[:8]])
    n_inputs = len(data[0])
    neuron = Neuron(get_starting_weights(n_inputs), get_starting_weights(1))
    for point in data:
        fires = neuron.activates(point)
        print("{} = {}".format(point, fires))


def main():
    run_iris()
    # run_diabetes()


if __name__ == "__main__":
    main()
