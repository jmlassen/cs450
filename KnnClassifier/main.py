from sklearn import datasets
from KnnClassifier.Driver import Driver


def main():
    driver = Driver()
    driver.run_iris_classification(datasets.load_iris())


if __name__ == '__main__':
    main()