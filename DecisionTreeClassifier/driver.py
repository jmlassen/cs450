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
        pass

    def run_voting(self):
        pass

    def _run_generic_test(self, data, target, name):
        print("Running classification test on {} data set".format(name))
        our_classifier = Id3DecisionTreeClassifier()
        our_results = cross_val_score(our_classifier, data, target, self.cv)
        print("Accuracy results from our classifier: {}".format(our_results))
        their_classifier = DecisionTreeClassifier()
        their_results = cross_val_score(their_classifier, data, target, self.cv)
        print("Accuracy results from their classifier: {}".format(their_results))
