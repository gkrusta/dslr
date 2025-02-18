import sys
import json
import pandas as pd
import numpy as np
from toolkit import DataParser

class LogisticRegression():
    def __init__(self):
        self.data = None
        self.lr = 0.1
        self.iterations = 2000
        self.weights = {}
        self.houses = []
        self.mean = {}
        self.std = {}


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def compute_cost(self, y_label, y_predicted):
        epsilon = 1e-9
        m = len(y_label)
        cost = 1 / m * np.sum(y_label * np.log(y_predicted + epsilon) + (1 - y_label) * np.log(1 - y_predicted + epsilon))
        return cost


    def gradient(self, x, y_label, y_predicted):
        m = len(y_label)
        grad = 1 / m * np.dot(x.T, (y_predicted - y_label)) #multiplicación de matrices
        return grad


    def parse_arguments(self, dataset):
        all_data = DataParser.open_file(dataset)
        self.houses = all_data["Hogwarts House"].unique().tolist()
        labels = {}
        for house in self.houses:
            labels[house] = (all_data['Hogwarts House'] == house).astype(int)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Defense Against the Dark Arts', 'Hogwarts House']
        data = all_data.drop(columns=columns_to_drop)
        self.data =  DataParser.replace_nan_values(data)
        self.weights = {house: [] for house in self.houses}

        for house, label in labels.items():
            self.data[f'{house}_label'] = label


    def standardize(self):
        for col in self.data.columns:
            if isinstance(self.data[col].iloc[0], int) or isinstance(self.data[col].iloc[0], float):
                self.mean[col] =  np.sum(self.data[col]) / len(self.data[col]) 
                value = 0.0
                for num in self.data[col]:
                    value += (num - self.mean[col]) ** 2
                self.std[col] = (value / len(self.data[col])) ** 0.5
                self.data[col] = (self.data[col] - self.mean[col]) / self.std[col]


    def calculate_weights(self):
        data_wo_label = self.data.iloc[:,:-4]
        X = np.insert(data_wo_label.values, 0, 1, axis=1)
        for house in self.houses:
            theta = np.zeros(X.shape[1], dtype=float)
            for _ in range(self.iterations):
                z = X.dot(theta)
                h = self.sigmoid(z)
                cost = self.compute_cost(self.data[f"{house}_label"], h)
                grad = self.gradient(X, self.data[f"{house}_label"], h)
                theta = theta - self.lr * grad
            self.weights[house] = theta.tolist()
    

    def destandarize(self):
        data_wo_label = self.data.iloc[:,:-4]
        final_weights = {}
        for house in self.houses:
            theta = np.array(self.weights[house])
            bias = theta[0]
            weights = []
            for i, col in enumerate(data_wo_label):
                weights.append(float(theta[i+1] / self.std[col]))
            bias_final = bias - sum(theta[i+1] * self.mean[col] / self.std[col] for i, col in enumerate(data_wo_label))
            final_weights[house] = {"bias": bias_final, "weights": weights}
        return final_weights


    def data_file(self):
        try:
            final_weights = self.destandarize()
            with open("weights.json", "w") as file:
                json.dump(final_weights, file, indent=4)
        except Exception as e:
            print(e)
            sys.exit(1)
        

def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./logreg_train.py dataset_name")
        sys.exit(1)

    lr = LogisticRegression()
    lr.parse_arguments(sys.argv[1])
    lr.standardize()
    lr.calculate_weights()
    lr.data_file()


if __name__ == "__main__":
    main()