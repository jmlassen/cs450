import csv
from sklearn import datasets
import numpy as np
from NeuralNetwork.neural_network import NeuralNetwork
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
    target = []
    # Read data
    with open('pima-indians-diabetes.data') as diabetes_file:
        diabetes_reader = csv.reader(diabetes_file, quoting=csv.QUOTE_NONNUMERIC)
        for row in diabetes_reader:
            data.append(row[:8])
            target.append(int(row[8]))
    n_inputs = len(data[0])
    n_outputs = len(set(target))
    network = NeuralNetwork()
    network.create_network(n_inputs, n_outputs, (2, 3))
    print("Neural network results accuracy: {}".format(cross_val_score(network, np.array(data), np.array(target), 3)))


def main():
    run_iris()
    run_diabetes()
#     data visualization:
#     matplotlib
#     bokeh


if __name__ == "__main__":
    main()
