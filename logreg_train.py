import sys
import pandas as pd
import numpy as np
from toolkit import DataParser

class LogisticRegression():
    def __init__(self):
        self.data = None
        self.lr = 0.001
        self.iterations = 1000
        self.weights = {}
        self.bias = 0


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def compute_cost(self, y_label, y_predicted):
        epsilon = 1e-9
        m = len(y_label)
        cost = 1 / m * np.sum(y_label * np.log(y_predicted + epsilon) + (1 - y_label) * np.log(1 - y_predicted + epsilon))
        return cost
    
    def gradient(self):
        pass


    def parse_arguments(self, dataset):
        all_data = DataParser.open_file(dataset)
        houses = all_data["Hogwarts House"].unique()
        labels = {}
        for house in houses:
            labels[house] = (all_data['Hogwarts House'] == house).astype(int)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Defense Against the Dark Arts', 'Hogwarts House']
        data = all_data.drop(columns=columns_to_drop)
        self.data =  DataParser.replace_nan_values(data)
        self.weights = {house: [] for house in houses}

        for house, label in labels.items():
            self.data[f'{house}_label'] = label
        
        print(self.data.head())


    def standardize(self):
        for col in self.data.columns:
            if isinstance(self.data[col].iloc[0], int) or isinstance(self.data[col].iloc[0], float):
                mean =  np.sum(self.data[col]) / len(self.data[col]) 
                value = 0.0
                for num in self.data[col]:
                    value += (num - mean) ** 2
                std = (value / len(self.data[col])) ** 0.5
                self.data[col] = (self.data[col] - mean) / std

    def calculate_weights(self):
        #for iteraciones
        data_wo_label = self.data.iloc[:,:-4]
        theta = np.zeros(len(data_wo_label.columns), dtype=int)
        z = data_wo_label.dot(theta)
        h = self.sigmoid(z)
        cost = self.compute_cost(self.data["Ravenclaw_label"], h)
        #gradiente

def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./logreg_train.py dataset_name")
        sys.exit(1)

    lr = LogisticRegression()
    lr.parse_arguments(sys.argv[1])
    lr.standardize()
    lr.calculate_weights()


if __name__ == "__main__":
    main()