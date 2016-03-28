from EnsembleLearning.driver import Driver


def main():
    driver = Driver()
    driver.run_abalone()
    driver.run_lenses()
    driver.run_letter_recognition()
    driver.run_mushroom()


if __name__ == '__main__':
    main()