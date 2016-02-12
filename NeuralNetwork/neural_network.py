from NeuralNetwork.neuron import Neuron
from NeuralNetwork.utilities import get_starting_weights


class NeuralNetwork:
    def __init__(self):
        self.network = []
        self.n_inputs = 0

    def create_network(self, n_inputs, n_outputs, hidden_layers):
        # Check to make sure inputs and outputs are greater than 0.
        if n_inputs < 1 or n_outputs < 1:
            raise Exception("n_inputs and n_outputs must be greater than 0!")
        # Create each hidden layer
        for layer in hidden_layers:
            self._create_layer(layer)
        # Create output layer
        self._create_layer(n_outputs)

    def fit(self, data, target):
        # Check data first.
        self._check_data(data)

    def predict(self, data):
        # Check data first.
        self._check_data(data)

    def _check_data(self, data):
        """Checks whether we are ready to run fit or predict on our network.

        :param data:
        :return:
        """
        if len(self.network) == 0 or self.n_inputs == 0:
            raise Exception("Network must be created before calling fit or predict.")
        if len(data[0]) != self.n_inputs:
            raise Exception("Data must have same number of fields as n_inputs when network created.")

    def _create_layer(self, n_neurons):
        """Creates a layer in the network.

        :param n_neurons:
        :return:
        """
        # Check n_neurons is greater than 0:
        if n_neurons < 1:
            raise Exception("Layer must have at least 1 node!")
        # Create a new neuron layer.
        layer = []
        # Determine number of inputs needed.
        if len(self.network) == 0:
            # If the network is empty, use create_network n_inputs
            n_inputs = self.n_inputs
        else:
            # Get n_inputs from the length of the last array put into the network array.
            n_inputs = len(self.network[-1])
        for i in range(n_neurons):
            # Create neuron, you should rethink the way you generate starting weights.
            # Append the neuron to the layer
            layer.append(Neuron(get_starting_weights(n_inputs), get_starting_weights(1)))
        # Append layer to the network.
        self.network.append(layer)
