import csv
from KnnClassifier.car import Car


class CarProcessor:
    """Load and converts the Car data set from file.

    """
    def load_car(self, file_name='car.data'):
        data, target = self._read_car_dataset(file_name)
        self._convert_nominal_to_numeric(data, 0, ['low', 'med', 'high', 'vhigh'])
        self._convert_nominal_to_numeric(data, 1, ['low', 'med', 'high', 'vhigh'])
        self._convert_nominal_to_numeric(data, 2, ['2', '3', '4', '5more'])
        self._convert_nominal_to_numeric(data, 3, ['2', '4', 'more'])
        self._convert_nominal_to_numeric(data, 4, ['small', 'med', 'big'])
        self._convert_nominal_to_numeric(data, 5, ['low', 'med', 'high'])
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

    def _convert_nominal_to_numeric(self, data, col_index, values):
        for i in range(len(data)):
            data[i][col_index] = values.index(data[i][col_index])
