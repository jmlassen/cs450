from sklearn import datasets
from KnnClassifier.driver import Driver


def main():
    """Calls run methods in driver class.
    """
    driver = Driver()
    driver.run_iris_classification(datasets.load_iris())
    driver.run_car_classification('car.data')


if __name__ == '__main__':
    main()
