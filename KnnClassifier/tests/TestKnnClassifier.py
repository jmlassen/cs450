import unittest
from KnnClassifier.knn_classifier import KnnClassifier


class TestKnnClassifier(unittest.TestCase):
    def test_classifier_can_predict(self):
        k = KnnClassifier()


if __name__ == '__main__':
    unittest.main()
