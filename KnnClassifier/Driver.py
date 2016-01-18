from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from KnnClassifier.knn_classifier import KnnClassifier
from KnnClassifier.knn_classifier_tester import KnnClassifierTester

TRAINING_DATA_PERCENTAGE = .7
K = 3


class Driver:
    def __init__(self):
        pass

    def run_iris_classification(self, iris):
        """Tests our KNN classifier.

        :param iris: The iris dataset from sklearn library.
        :return:
        """
        # Consistently shuffle the data and target
        data, target = shuffle(iris.data, iris.target)
        # Calculate the number of records that will be used to train the classifier
        training_record_count = len(data) * TRAINING_DATA_PERCENTAGE
        # Create tester
        tester = KnnClassifierTester()

        # Create classifier and train
        classifier = KnnClassifier(n_neighbors=K)
        classifier.fit(data[:training_record_count], target[:training_record_count])
        results = tester.test(classifier, data[training_record_count:], target[training_record_count:])
        # Output results
        print("Our KnnClassifier Results on Iris DB: {}% accuracy".format(round(results * 100, 3)))

        # Test their KnnClassifier
        their_classifier = KNeighborsClassifier(n_neighbors=K)
        their_classifier.fit(data[:training_record_count], target[:training_record_count])
        results = tester.test(their_classifier, data[training_record_count:], target[training_record_count:])
        print("Their KNeighborsClassifier Results on Iris DB: {}% accuracy".format(round(results * 100, 3)))
