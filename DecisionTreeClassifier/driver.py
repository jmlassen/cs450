import csv

import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from DecisionTreeClassifier.id3_decision_tree_classifier import Id3DecisionTreeClassifier
from utilities import cross_val_score


class Driver:
    def __init__(self, cv=3):
        self.cv = cv

    def run_iris(self):
        iris = load_iris()
        self._run_generic_test(iris.data, iris.target, 'Iris')

    def run_lenses(self):
        data = []
        target = []
        with open('lenses.data')as file:
            file_reader = csv.reader(file, delimiter=' ')
            for row in file_reader:
                data.append([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
                target.append(int(row[5]))
        self._run_generic_test(np.array(data), np.array(target), 'Lenses')

    def run_voting(self):
        data = []
        target = []
        class_names = {'democrat': 0, 'republican': 1}
        answers = {'y': 0, 'n': 1, '?': 2}
        with open('house-votes-84.data') as file:
            file_reader = csv.reader(file)
            for row in file_reader:
                target.append(class_names[row[0]])
                point = []
                for i in range(1, len(row)):
                    point.append(answers[row[i]])
                data.append(point)
        self._run_generic_test(np.array(data), np.array(target), 'Voting')

    def _run_generic_test(self, data, target, name):
        print("Running classification test on {} data set".format(name))
        our_classifier = Id3DecisionTreeClassifier()
        our_results = cross_val_score(our_classifier, data, target, self.cv)
        print("Accuracy results from our classifier: {}".format(our_results))
        their_classifier = DecisionTreeClassifier()
        their_results = cross_val_score(their_classifier, data, target, self.cv)
        print("Accuracy results from their classifier: {}".format(their_results))
