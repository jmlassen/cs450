import logging

import numpy as np
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from DecisionTreeClassifier.bak.id3_decision_tree_classifier import Id3DecisionTreeClassifier


class Driver:
    def __init__(self, k_folds=3):
        self.k_folds = k_folds
        self.logger = logging.getLogger(Driver.__name__)
        self.logger.debug('Initialized new class object.')

    def run_iris_classification(self, database):
        self.logger.debug('Started run_iris_classification function.')
        self._run_generic_classification('Iris', database.data, database.target)

    def run_lenses_classification(self):
        pass

    def run_voting_classification(self):
        pass

    def _run_generic_classification(self, name, data, target):
        self.logger.debug('Started _run_generic_classification function.')
        if len(data) != len(target):
            self.logger.critical('Lengths of data and target lists do not match! There is no more we can do!')
            raise Exception('Length of data and target lists inconsistent.')
        if len(data) % self.k_folds > 0:
            self.logger.warn(
                    "{} does not divide evenly into {}, this means some fold will be disproportionate."
                    .format(self.k_folds, len(data)))
        # Test with our benchmark classifier.
        self.logger.debug('Creating sklean DecisionTreeClassifier, setting random state to 0.')
        their_classifier = DecisionTreeClassifier(random_state=0)
        self.logger.debug("Calling cross_val_score with {} folds.".format(self.k_folds))
        results = cross_validation.cross_val_score(their_classifier, data, target, cv=self.k_folds)
        self.logger.debug("Results from cross_val_score function call: {}".format(results))
        self.logger.debug("Average accuracy of their classifier found to be {}.".format(results.mean()))
        # Output results to command line.
        print("For {} test, their classifier average found to be {}".format(name, results.mean()))
        # Start testing our classifier
        self.logger.debug('Creating our Id3DecisionTreeClassifier.')
        our_classifier = Id3DecisionTreeClassifier()
        results = self._cross_val_score(our_classifier, data, target, cv=self.k_folds)
        self.logger.debug("Results from our cross_val_score function call: {}".format(results))
        self.logger.debug("Average accuracy of our classifier found to be {}.".format(results.mean()))
        print("For {} test, our classifier average found to be {}.".format(name, results.mean()))

    def _cross_val_score(self, classifier, data, target, cv):
        self.logger.debug('Started _cross_val_score function.')
        self.logger.debug('Shuffling data and target arrays.')
        data, target = shuffle(data, target)
        fold_len = int(len(data) / cv)
        self.logger.debug("Fold length calculated to be {}.".format(fold_len))
        results = []
        for i in range(cv):
            self.logger.debug("Working on {} fold.".format(i))
            start_index = fold_len * i
            end_index = fold_len * (i + 1)
            train_data = np.concatenate((data[:start_index], data[end_index:]), axis=0)
            train_target = np.concatenate((target[:start_index], target[end_index:]), axis=0)
            self.logger.debug("Size of training data found to be {}.".format(len(train_data)))
            self.logger.debug("Sending training data to our classifier.")
            classifier.fit(train_data, train_target)
            self.logger.debug("Training finished, sending testing data to classifier.")
            self.logger.debug("Range for testing data found to be {} - {}.".format(start_index, end_index - 1))
            prediction = classifier.predict(data[start_index:end_index])
            if len(prediction) != len(target[start_index:end_index]):
                self.logger.critical('Prediction results length does not match target results!')
                raise Exception('Classifier predict function did not return correct number of results!')
            accuracy = np.sum(prediction == target[start_index:end_index]) / len(prediction)
            self.logger.debug("Accuracy found to be {}.".format(accuracy))
            results.append(accuracy)
        return np.array(results)
