import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from NeuralNetwork.neuron import Neuron
from NeuralNetwork.utilities import *

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_VALIDATION_PERCENTAGE = 0.2
DEFAULT_EPOCHS = 500


class NeuralNetwork:
    """Object to hold a neural network.

    """

    def __init__(self, learning_rate=DEFAULT_LEARNING_RATE, validation_percentage=DEFAULT_VALIDATION_PERCENTAGE,
                 epochs=DEFAULT_EPOCHS):
        """Initializes our network.

        Initializes container holding network nodes to an empty array. Sets n_inputs to 0, since we will be using this
        to check if we can run fit or predict later.

        :return:
        """
        self.learning_rate = learning_rate
        self.validation_percentage = validation_percentage
        self.network = []
        self.n_inputs = 0
        self.epochs = 500

    def create_network(self, n_inputs, n_outputs, hidden_layers, weights=None):
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
            self._create_layer(layer, weights)
        # Create output layer
        self._create_layer(n_outputs, weights)

    def fit(self, data, target):
        self._check_network_created(data)
        validation_split = int(len(data) * self.validation_percentage)
        epoch_accuracy = []
        for _ in range(self.epochs):
            for i in range(len(data[:validation_split])):
                results = self._feed_forward(data[i])
                updated_weights = self._back_propagation(results, target[i])
                self._apply_updated_weights(updated_weights)
            prediction = self.predict(data[validation_split:])
            epoch_accuracy.append(accuracy_score(target[validation_split:], prediction))
        self._plot(epoch_accuracy)

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

    def _create_layer(self, n_neurons, weights=None):
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
            # TEMP FOR TESTING
            if weights is None:
                temp_weights = get_starting_weights(n_inputs + 1)
            else:
                temp_weights = weights[i][:(n_inputs + 1)]
            # Create neuron, you should rethink the way you generate starting weights.
            # Append the neuron to the layer
            layer.append(Neuron(temp_weights))
        # Append layer to the network.
        self.network.append(layer)

    def _feed_forward(self, point):
        results = [point]
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

        The first element in target is the original inputs. The last element is the output from the network.

        :param results:
        :param target:
        :return: Updated weights for nodes.[:network [:layer [:node_weights]]]
        """
        # Create an array modeled after our network, since that makes more sense than pushing back right?
        updated_weights = np.empty_like(self.network)
        delta = self._calculate_network_delta_values(results, target)
        # print_network(self.network)
        # print_network(results)
        for layer in range(len(self.network)):
            updated_weights[layer] = np.empty_like(self.network[layer])
            for neuron in range(len(self.network[layer])):
                updated_weights[layer][neuron] = np.empty_like(self.network[layer][neuron].weights)
                # Update bias node
                updated_weights[layer][neuron][0] = calc_updated_weight(
                    self.network[layer][neuron].weights[0],
                    self.learning_rate,
                    delta[layer][neuron],
                    self.network[layer][neuron].bias
                )
                for weight in range(1, len(self.network[layer][neuron].weights)):
                    updated_weights[layer][neuron][weight] = calc_updated_weight(
                        self.network[layer][neuron].weights[weight],
                        self.learning_rate,
                        delta[layer][neuron],
                        results[layer][weight - 1]
                    )
        return updated_weights

    def _calculate_network_delta_values(self, results, target):
        # print_network(self.network)
        # print_network(results)
        delta = np.empty_like(self.network)
        delta[-1] = np.empty(len(self.network[-1]))
        # Find delta values for last layer
        for i in range(len(results[-1])):
            node_target = 1 if i == target else 0
            delta[-1][i] = calc_output_delta(results[-1][i], node_target)
        # Step continue backwards through our network
        for i in reversed(range(len(results[:-2]))):
            delta[i] = np.empty_like(self.network[i], dtype=float)
            for j in range(len(self.network[i])):
                w = []
                for neuron in self.network[i + 1]:
                    w.append(neuron.weights[j + 1])
                result = results[i + 1][j]
                delta_val = delta[i + 1]
                delta[i][j] = calc_hidden_delta(result, w, delta_val)
        return delta

    def _apply_updated_weights(self, updated_weights):
        for layer in range(len(updated_weights)):
            for neuron in range(len(updated_weights[layer])):
                for weight in range(len(updated_weights[layer][neuron])):
                    self.network[layer][neuron].weights[weight] = updated_weights[layer][neuron][weight]

    def _plot(self, epoch_accuracy):
        x = range(self.epochs)
        plt.scatter(x, epoch_accuracy)
        plt.show()
