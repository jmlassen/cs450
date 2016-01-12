import numpy as np
from sklearn import datasets

from HardCodedClassifier.HardCoded import HardCoded


def randomize_dataset(dataset):
    # manually set the seed
    np.random.seed(1)
    np.random.shuffle(dataset.data)
    # reset the seed
    np.random.seed(1)
    np.random.shuffle(dataset.target)


def test_classifier(classifier, dataset, train_data_count):
    accuracy_count = 0
    for i in range(train_data_count):
        if classifier.predict(dataset.data[i]) == dataset.target[i]:
            accuracy_count += 1
    return accuracy_count / train_data_count


def train_classifier(classifier, dataset):
    test_data_count = int(len(dataset.data) * .3)
    classifier.train(dataset.data[test_data_count:-1], dataset.target[test_data_count:-1])
    return test_data_count


def main():
    iris = datasets.load_iris()
    randomize_dataset(iris)
    classifier = HardCoded()
    train_data_count = train_classifier(classifier, iris)
    accuracy = test_classifier(classifier, iris, train_data_count)
    print('Classifier Accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
