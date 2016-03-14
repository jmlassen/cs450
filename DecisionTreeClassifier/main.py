from DecisionTreeClassifier.driver import Driver


def main():
    driver = Driver()
    driver.run_iris()
    driver.run_lenses()
    driver.run_voting()


if __name__ == '__main__':
    main()
