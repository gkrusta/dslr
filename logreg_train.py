import sys
import pandas as pd
import numpy as np
from toolkit import DataParser

class Logisticregression():
    def __init__(self):
        self.data = None
        self.lr = 0.001
        self.iterations = 1000
        self.weights = 0
        self.bias = 0


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def compute_cost(self, y_label, y_predicted):
        epsilon = 1e-9
        pass


    def parse_arguments(self, dataset):
        all_data = DataParser.open_file(dataset)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Defense Against the Dark Arts']
        data = all_data.drop(columns=columns_to_drop)
        self.data =  DataParser.replace_nan_values(data)
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        labels = {}
        for house in houses:
            labels[house] = (self.data['Hogwarts House'] == house).astype(int)

        for house, label in labels.items():
            self.data[f'{house}_label'] = label

        print(self.data.head())


def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./logreg_train.py dataset_name")
        sys.exit(1)

    lr = Logisticregression()
    lr.parse_arguments(sys.argv[1])


if __name__ == "__main__":
    main()