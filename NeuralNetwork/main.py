import csv
from sklearn import datasets

from NeuralNetwork.neuron import Neuron
from NeuralNetwork.utilities import get_starting_weights


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
