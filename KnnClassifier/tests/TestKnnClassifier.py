import unittest
from KnnClassifier.KnnClassifier import KnnClassifier

class TestKnnClassifier(unittest.TestCase):

    def test_classifier_can_predict(self):
        k = KnnClassifier()


if __name__ == '__main__':
    unittest.main()