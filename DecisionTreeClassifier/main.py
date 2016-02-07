from sklearn import datasets
from DecisionTreeClassifier.driver import Driver
import logging


def set_logging_settings():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)


def main():
    set_logging_settings()
    driver = Driver()
    driver.run_iris_classification(datasets.load_iris())
    driver.run_lenses_classification()
    driver.run_voting_classification()


if __name__ == '__main__':
    main()
