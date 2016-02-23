from unittest import TestCase

from NeuralNetwork.neural_network import NeuralNetwork


class TestNeuralNetwork(TestCase):
    def test__back_propagation(self):
        data = [[1, 1, 2]]
        target = [1, 0, 0]
        nn = NeuralNetwork()
        nn.create_network(len(data[0]), 2, (1,), weights=[[-.1, .2, .1, -.4], [-.15, -.2, .3]])
        r = nn._feed_forward(data[0])
        uw = nn._back_propagation(r, target[0])
