import copy

from NeuralNetwork.neuron import Neuron
from NeuralNetwork.utilities import get_starting_weights


class NeuralNetwork:
    """Object to hold a neural network.

    """

    def __init__(self):
        """Initializes our network.

        Initializes container holding network nodes to an empty array. Sets n_inputs to 0, since we will be using this
        to check if we can run fit or predict later.

        :return:
        """
        self.network = []
        self.n_inputs = 0

    def create_network(self, n_inputs, n_outputs, hidden_layers):
        """Creates our neural network.

        This must be run before fit or predict is called.

        :param n_inputs: Number of fields (columns) provided in data set.
        :param n_outputs: Number of different labels in a data set.
        :param hidden_layers: Tuple representing the layout of the hidden layers.
        :return: None
        """
        # Check for potential inputs in network creation
        self._check_network_creation_inputs(n_inputs, n_outputs, hidden_layers)
        # Save n_inputs for error checking later.
        self.n_inputs = n_inputs
        # Create each hidden layer
        for layer in hidden_layers:
            self._create_layer(layer)
        # Create output layer
        self._create_layer(n_outputs)

    def fit(self, data, target):
        # Check data first.
        self._check_network_created(data)
        # Loop through each data point
        for i in range(len(data)):
            results = self._feed_forward(data[i])
            updated_weights = self._back_propagation(results, target[i])

    def predict(self, data):
        """Predicts a list of points.

        :param data:
        :return:
        """
        # Check data first.
        self._check_network_created(data)
        # Create container for results.
        results = []
        for point in data:
            # Run point through network and save result.
            feed_forward_results = self._feed_forward(point)
            results.append(feed_forward_results[-1].index(max(feed_forward_results[-1])))
        return results

    def _check_network_creation_inputs(self, n_inputs, n_outputs, hidden_layers):
        """Tests that must be performed on network creation inputs to make sure we can create a valid neural network.

        :param n_inputs: Number of inputs that will be provided for each point of data
        :param n_outputs: Number of possible categories
        :param hidden_layers: Tuple representing the number of hidden layers
        :return: None
        """
        # Check to make sure inputs and outputs are greater than 0.
        if n_inputs < 1:
            raise Exception("n_inputs must be greater than 0!")
        elif n_outputs < 1:
            raise Exception("n_outputs must be greater than 0!")
        for n_neurons in hidden_layers:
            if n_neurons < 1:
                raise Exception("Layer must have at least 1 node!")

    def _check_network_created(self, data):
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
            weights = get_starting_weights(n_inputs + 1)
            # Create neuron, you should rethink the way you generate starting weights.
            # Append the neuron to the layer
            layer.append(Neuron(weights[:n_inputs], weights[-1]))
        # Append layer to the network.
        self.network.append(layer)

    def _feed_forward(self, point):
        results = [copy.deepcopy(point)]
        # Loop through each layer
        for n_layer in range(len(self.network)):
            results.append(self._run_inputs_through_layer(results[-1], self.network[n_layer]))
        # Return the largest value found
        return results

    def _run_inputs_through_layer(self, inputs, layer):
        results = []
        for neuron in layer:
            results.append(neuron.activates(inputs))
        return results

    def _back_propagation(self, results, target):
        """Run back propagation with the results from running a point through the network.

        :param results:
        :param target:
        :return:
        """
        # Loop through all but last layer of results in reverse.
        for layer in reversed(results[:-1]):
            pass
        return []
