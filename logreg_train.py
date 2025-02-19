import sys
import pandas as pd
import numpy as np
from toolkit import DataParser
import json

class LogisticRegression():
    def __init__(self):
        self.data = None
        self.features = None
        self.houses = None
        self.feature_count = 0
        self.lr = 0.001
        self.iterations = 1000
        self.weights = {}
        self.bias = {}


    def sigmoid(self, x):
        """"
        Sigmoid is a mathematical function that maps any real-valued number into a value
        between 0 and 1 to convert outputs into probabilities.
        """
        return 1 / (1 + np.exp(-x))


    def compute_cost(self, y_label, y_predicted):
        epsilon = 1e-9
        pass


    def parse_arguments(self, dataset):
        all_data = DataParser.open_file(dataset)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Defense Against the Dark Arts']
        data = all_data.drop(columns=columns_to_drop)
        self.data =  DataParser.replace_nan_values(data)
        self.features = self.data.copy()
        self.features = self.features.drop(columns=['Hogwarts House'])
        self.houses = all_data["Hogwarts House"].unique()
        labels = {}
        self.feature_count = len(self.data.columns) - 1
        for house in self.houses:
            labels[house] = (self.data['Hogwarts House'] == house).astype(int)
            self.weights[house] = np.zeros(self.feature_count)
            self.bias[house] = 0

        for house, label in labels.items():
            self.data[f'{house}_label'] = label
        print(self.data)


    def standardize(self):
        for col in self.data.columns:
            if isinstance(self.data[col].iloc[0], int) or isinstance(self.data[col].iloc[0], float):
                mean =  np.sum(self.data[col]) / len(self.data[col]) 
                value = 0.0
                for num in self.data[col]:
                    value += (num - mean) ** 2
                std = (value / len(self.data[col])) ** 0.5
                self.data[col] = (self.data[col] - mean) / std

    def train(self):
        """
        Trains a logistic regression classifier for each Hogwarts house using gradient descent.
        """
        for house in self.weights.keys():
            y_label = self.data[f'{house}_label'].values
            for i in range(self.iterations):
                # prediction
                z = np.dot(self.features, self.weights[house]) + self.bias[house]
                y_pred = self.sigmoid(z)
                # error
                # partial derivative for each weight and bias 
                for i in range(self.feature_count):
                    self.weights[house][i] =  
                self.bias[house] = 

    def save_weights(self, filename="weights.json"):
        """
        Saves the trained weights and biases to a JSON file.
        """
        data = {
            "weights": {house: self.weights[house].tolist() for house in self.weights},
            "bias": self.bias
        }
        with open(filename, "w") as file:
            file.dump(data, file) #indent=4


def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./logreg_train.py dataset_name")
        sys.exit(1)

    lr = LogisticRegression()
    lr.parse_arguments(sys.argv[1])
    lr.standardize()
    lr.train()
    lr.save_weights()


if __name__ == "__main__":
    main()