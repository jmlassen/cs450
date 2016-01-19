import csv
from KnnClassifier._car import Car


class CarProcessor:
    """Load and converts the Car data set from file.

    """
    def load_car(self, file_name='car.data'):
        data, target = self._read_car_dataset(file_name)

        return Car(data, target)

    def _read_car_dataset(self, file_name):
        data, target = [], []
        with open(file_name) as data_file:
            car_data = csv.reader(data_file, delimiter=',')
            for row in car_data:
                # Append the training data
                data.append(row[:6])
                # Append the label
                target.append(row[6])
        return data, target

    def _convert_nominal_to_numeric(self, data, index, values):
        pass
