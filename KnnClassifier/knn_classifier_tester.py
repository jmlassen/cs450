import numpy as np


class KnnClassifierTester:
    """Runs tests on a KnnClassifier object.
    """

    def test(self, classifier, test_data, test_target):
        """Tests KnnClassifier with passed testing data. Returns accuracy, maybe more?

        :param classifier:
        :param test_data:
        :param test_target:
        :return:
        """
        results = classifier.predict(test_data)
        # Make sure we have enough answers for predictions
        assert len(results) == len(test_target)
        return np.sum(results == test_target) / len(results)
