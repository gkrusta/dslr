import sys
import pandas as pd
import numpy as np
from toolkit import DataParser

class LogisticRegression():
    def __init__(self):
        self.data = None
        self.lr = 0.001
        self.iterations = 1000
        self.weights = []
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
        houses = all_data["Hogwarts House"].unique()
        labels = {}
        for house in houses:
            labels[house] = (self.data['Hogwarts House'] == house).astype(int)

        for house, label in labels.items():
            self.data[f'{house}_label'] = label


    def standardize(self):
        for col in self.data.columns:
            if isinstance(self.data[col].iloc[0], int) or isinstance(self.data[col].iloc[0], float):
                mean =  np.sum(self.data[col]) / len(self.data[col]) 
                value = 0.0
                for num in self.data[col]:
                    value += (num - mean) ** 2
                std = (value / len(self.data[col])) ** 0.5
                self.data[col] = (self.data[col] - mean) / std


def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./logreg_train.py dataset_name")
        sys.exit(1)

    lr = LogisticRegression()
    lr.parse_arguments(sys.argv[1])
    lr.standardize()


if __name__ == "__main__":
    main()