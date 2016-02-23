from unittest import TestCase

from NeuralNetwork.utilities import sigmoid


class TestSigmoid(TestCase):
    def test_sigmoid(self):
        print(sigmoid(-.4))
