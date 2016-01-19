from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from KnnClassifier.knn_classifier import KnnClassifier
from KnnClassifier.knn_classifier_tester import KnnClassifierTester

TRAINING_DATA_PERCENTAGE = .7
K = 3


class Driver:
    def __init__(self):
        """Initializes the driver class.

        This should probably set the global variables... but it's not.

        :return: None
        """
        pass

    def run_iris_classification(self, iris):
        """Runs an classification test on the Iris data set.

        :param iris: The Iris data set from the sklearn datasets.
        :return: None
        """
        self._run_generic_classification(iris.data, iris.target, 'Iris')

    def run_car_classification(self, car):
        self._run_generic_classification(car.data, car.target, 'Car')

    def _run_generic_classification(self, sorted_data, sorted_target, test_name):
        data, target = shuffle(sorted_data, sorted_target)
        # Calculate the number of records that will be used to train the classifier
        training_record_count = len(data) * TRAINING_DATA_PERCENTAGE
        # Create tester
        tester = KnnClassifierTester()

        # Create classifier and train
        classifier = KnnClassifier(n_neighbors=K)
        classifier.fit(data[:training_record_count], target[:training_record_count])
        results = tester.test(classifier, data[training_record_count:], target[training_record_count:])
        # Output results
        print("Our {} KNN Classifier Results: {}% accuracy".format(test_name, round(results * 100, 3)))

        # Test their KnnClassifier
        their_classifier = KNeighborsClassifier(n_neighbors=K)
        their_classifier.fit(data[:training_record_count], target[:training_record_count])
        results = tester.test(their_classifier, data[training_record_count:], target[training_record_count:])
        print("Their {} KNN Classifier Results: {}% accuracy".format(test_name, round(results * 100, 3)))




