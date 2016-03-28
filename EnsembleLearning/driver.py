import csv
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

DEFAULT_TESTING_PERCENTAGE = 0.7


class Driver:
    def __init__(self, testing_percentage=DEFAULT_TESTING_PERCENTAGE):
        self.testing_percentage = testing_percentage

    def run_abalone(self):
        print("Running abalone database:")
        data = []
        target = []
        with open('abalone.data') as file:
            file_reader = csv.reader(file)
            for file_reader_row in file_reader:
                target.append(file_reader_row[0])
                data.append(file_reader_row[1:9])
        self._run_generic(np.array(data), np.array(target))

    def run_lenses(self):
        print("Running lenses database:")
        data = []
        target = []
        with open('lenses.data') as file:
            file_reader = csv.reader(file, delimiter=' ')
            for file_reader_row in file_reader:
                target.append(file_reader_row[1])
                data.append([int(i) for i in file_reader_row[2:5]])
        self._run_generic(np.array(data), np.array(target))

    def run_letter_recognition(self):
        print("Running letter recognition database:")
        data = []
        target = []
        with open('letter-recognition.data') as file:
            file_reader = csv.reader(file)
            for file_reader_row in file_reader:
                target.append(file_reader_row[0])
                data.append([int(i) for i in file_reader_row[1:17]])
        self._run_generic(np.array(data), np.array(target))

    def run_mushroom(self):
        print("Running mushroom database:")
        data = []
        target = []
        with open('agaricus-lepiota.data') as file:
            file_reader = csv.reader(file)
            for file_reader_row in file_reader:
                target.append(file_reader_row[0])
                data.append([ord(i) for i in file_reader_row[1:23]])
        self._run_generic(np.array(data), np.array(target))

    def _run_generic(self, data, target):
        shuffled_data, shuffled_target = shuffle(data, target)
        testing_record_count = int(len(shuffled_data) * self.testing_percentage)
        self._run_benchmark_tests(shuffled_data, shuffled_target, testing_record_count)
        self._run_bagging_test(shuffled_data, shuffled_target, testing_record_count)
        self._run_ada_boost_test(shuffled_data, shuffled_target, testing_record_count)
        self._run_random_forrest_test(shuffled_data, shuffled_target, testing_record_count)
        print()

    def _run_benchmark_tests(self, data, target, testing_record_count):
        print("Running benchmark classifiers:")
        clfs = [KNeighborsClassifier(), DecisionTreeClassifier(), SVC()]
        for clf in clfs:
            accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
            print("{} accuracy: {}".format(type(clf).__name__, accuracy))

    def _run_bagging_test(self, data, target, testing_record_count):
        clf = BaggingClassifier()
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Bagging test with default parameters: {}".format(accuracy))
        clf = BaggingClassifier(bootstrap=False)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Bagging test without using bootstrap: {}".format(accuracy))
        clf = BaggingClassifier(bootstrap_features=True)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Bagging test using bootstrap_features: {}".format(accuracy))
        clf = BaggingClassifier(warm_start=True)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Bagging test using warm_start: {}".format(accuracy))

    def _run_generic_classification_test(self, clf, data, target, testing_record_count):
        clf.fit(data[:testing_record_count], target[:testing_record_count])
        predicted = clf.predict(data[testing_record_count:])
        return accuracy_score(target[testing_record_count:], predicted)

    def _run_ada_boost_test(self, data, target, testing_record_count):
        clf = AdaBoostClassifier()
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Ada boost test with default parameters: {}".format(accuracy))
        clf = AdaBoostClassifier(n_estimators=100)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Ada boost test with n_estimators=100: {}".format(accuracy))
        clf = AdaBoostClassifier(n_estimators=1000)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Ada boost test with n_estimators=1000: {}".format(accuracy))
        clf = AdaBoostClassifier(learning_rate=0.1)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Ada boost test with learning_rate=0.1: {}".format(accuracy))
        clf = AdaBoostClassifier(learning_rate=0.01)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Ada boost test with learning_rate=0.01: {}".format(accuracy))
        clf = AdaBoostClassifier(learning_rate=5)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Ada boost test with learning_rate=5: {}".format(accuracy))

    def _run_random_forrest_test(self, data, target, testing_record_count):
        clf = RandomForestClassifier()
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Random forest test with default parameters: {}".format(accuracy))
        clf = RandomForestClassifier(n_estimators=5)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Random forest test with n_estimators=5: {}".format(accuracy))
        clf = RandomForestClassifier(n_estimators=20)
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Random forest test with n_estimators=20: {}".format(accuracy))
        clf = RandomForestClassifier(criterion="entropy")
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Random forest test with criterion=entropy: {}".format(accuracy))
        clf = RandomForestClassifier(max_features="log2")
        accuracy = self._run_generic_classification_test(clf, data, target, testing_record_count)
        print("Random forest test with max_features=log2: {}".format(accuracy))
