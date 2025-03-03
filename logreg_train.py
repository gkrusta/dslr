import sys
import json
import pandas as pd
import numpy as np
from pyrsistent import optional
from toolkit import DataParser

class LogisticRegression():
    def __init__(self):
        self.data = None
        self.lr = 0.1
        self.iterations = 1000
        self.weights = {}
        self.bias = {}
        self.houses = []
        self.mean = {}
        self.std = {}


    def compute_cost(self, y_label, y_predicted):
        epsilon = 1e-9
        m = len(y_label)
        cost = 1 / m * np.sum(y_label * np.log(y_predicted + epsilon) + (1 - y_label) * np.log(1 - y_predicted + epsilon))
        return cost


    def weights_gradient(self, x, y_label, y_predicted):
        m = len(y_label)
        grad = 1 / m * np.dot(x.T, (y_predicted - y_label))
        return grad
    

    def bias_gradient(self, y_label, y_predicted):
        m = len(y_label)
        grad = 1 / m * np.sum(y_predicted - y_label)
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
        self.bias = {house: [] for house in self.houses}

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
        for house in self.houses:
            weight = np.zeros(data_wo_label.shape[1], dtype=float)
            bias = 0
            for _ in range(self.iterations):
                z = data_wo_label.dot(weight) + bias
                h = DataParser.sigmoid(z)
                cost = self.compute_cost(self.data[f"{house}_label"], h)
                w_grad = self.weights_gradient(data_wo_label, self.data[f"{house}_label"], h)
                b_grad = self.bias_gradient(self.data[f"{house}_label"], h)
                weight = weight - self.lr * w_grad
                bias = bias - self.lr * b_grad
            self.weights[house] = weight.tolist()
            self.bias[house] = bias
    

    def mini_batch_weights(self):
        batch = 256
        m = len(self.data)
        if m < batch:
            batch = m
        data_wo_label = self.data.iloc[:,:-4]
        for house in self.houses:
            weight = np.zeros(data_wo_label.shape[1], dtype=float)
            bias = 0
            for _ in range(self.iterations):
                for i in range(0, m, batch):
                    X = data_wo_label[i:i + batch]
                    X_data = self.data[i:i + batch]
                    z = X.dot(weight) + bias
                    h = DataParser.sigmoid(z)
                    cost = self.compute_cost(X_data[f"{house}_label"], h)
                    w_grad = self.weights_gradient(X, X_data[f"{house}_label"], h)
                    b_grad = self.bias_gradient(X_data[f"{house}_label"], h)
                    weight = weight - self.lr * w_grad
                    bias = bias - self.lr * b_grad
            self.weights[house] = weight.tolist()
            self.bias[house] = bias 
    

    def calculate_sgd(self):
        data_wo_label = self.data.iloc[:,:-4]
        for house in self.houses:
            weight = np.zeros(data_wo_label.shape[1], dtype=float)
            bias = 0
            for i in range(self.iterations):
                shuffled_data = self.data.sample(frac=1).reset_index(drop=True)                    
                feature_row = shuffled_data.iloc[i, :-4].values
                y_label = shuffled_data[f"{house}_label"].iloc[i]
                z = np.dot(feature_row, weight) + bias
                h = DataParser.sigmoid(z)

                w_grad = (h - y_label) * feature_row
                b_grad = h - y_label
                weight = weight - self.lr * w_grad
                bias = bias - self.lr * b_grad
            self.weights[house] = weight.tolist()
            self.bias[house] = bias 


    def destandarize(self):
        data_wo_label = self.data.iloc[:,:-4]
        final_weights = {}
        for house in self.houses:
            theta = np.array(self.weights[house])
            bias = self.bias[house]
            weights = []
            bias_final = 0
            for i, col in enumerate(data_wo_label):
                weights.append(float(theta[i] / self.std[col]))
                bias_final += theta[i] * self.mean[col] / self.std[col]
            bias_final = bias - bias_final
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
        

def train(train_path, weights_path=optional, config_path=None, visualize=False, flag='-b'):
    lr = LogisticRegression()
    lr.parse_arguments(train_path)
    lr.standardize()
    if flag == '-b':
        lr.calculate_weights()
    elif flag == '-s':
        lr.calculate_sgd()
    elif flag == '-m':
        lr.mini_batch_weights()
    lr.data_file()


def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./logreg_train.py dataset_name flag")
        print("Flag options:\n-b or leave empty: Batch GD (default)\n-s: stochastic\n-m: mini-batch GD")
        sys.exit(1)
    flag = '-b'
    if (len(sys.argv) > 2 and (sys.argv[2] == '-s' or sys.argv[2] == '-m')):
        flag = sys.argv[2]
    train(train_path=sys.argv[1], flag=flag)


if __name__ == "__main__":
    main()
