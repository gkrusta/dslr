import sys
import json
import pandas as pd
import numpy as np
from pyrsistent import optional
from toolkit import DataParser

class LogisticRegression():
    """
    Implements logistic regression with support for batch, mini-batch, and stochastic gradient descent.

    This class allows training a logistic regression model on a dataset using different optimization 
    techniques and provides methods for standardizing data, computing gradients, and exporting model weights.
    """

    def __init__(self):
        """
        Inits the class and the variables.
        """
        self.data = None
        self.lr = 0.1
        self.iterations = 1000
        self.weights = {}
        self.bias = {}
        self.houses = []
        self.mean = {}
        self.std = {}


    def weights_gradient(self, x, y_label, y_predicted):
        """
        Computes the gradient of the weights for logistic regression.
        
        Parameters:
        x (pd.Series): Dataset
        y_label (pd.Series): Column with the truth label (0 or 1) for each example.
        y_predicted (pd.Series): Column with the predicted probabilities for each example.
        """
        m = len(y_label)
        grad = 1 / m * np.dot(x.T, (y_predicted - y_label))
        return grad
    

    def bias_gradient(self, y_label, y_predicted):
        """
        Computes the gradient of the bias term in logistic regression.
        
        Parameters:
        y_label (pd.Series): Column with the truth label (0 or 1) for each example.
        y_predicted (pd.Series): Column with the predicted probabilities for each example.
        """
        m = len(y_label)
        grad = 1 / m * np.sum(y_predicted - y_label)
        return grad


    def parse_arguments(self, dataset):
        """
        Loads the dataset, cleans the data, and creates binary labels for each house.

        This method reads the dataset, removes unnecessary columns, replaces NaN values, and generates a 
        label column for each Hogwarts house.

        Parameters:
        dataset (str): Path to dataset
        """
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
        """
        Standardizes the numerical features of the dataset using z-score normalization.
        """
        for col in self.data.columns:
            if isinstance(self.data[col].iloc[0], int) or isinstance(self.data[col].iloc[0], float):
                self.mean[col] =  np.sum(self.data[col]) / len(self.data[col]) 
                value = 0.0
                for num in self.data[col]:
                    value += (num - self.mean[col]) ** 2
                self.std[col] = (value / len(self.data[col])) ** 0.5
                self.data[col] = (self.data[col] - self.mean[col]) / self.std[col]


    def calculate_weights(self):
        """
        Trains the logistic regression model using Batch Gradient Descent.
        """
        data_wo_label = self.data.iloc[:,:-4]
        for house in self.houses:
            weight = np.zeros(data_wo_label.shape[1], dtype=float)
            bias = 0
            for _ in range(self.iterations):
                z = data_wo_label.dot(weight) + bias
                h = DataParser.sigmoid(z)
                w_grad = self.weights_gradient(data_wo_label, self.data[f"{house}_label"], h)
                b_grad = self.bias_gradient(self.data[f"{house}_label"], h)
                weight = weight - self.lr * w_grad
                bias = bias - self.lr * b_grad
            self.weights[house] = weight.tolist()
            self.bias[house] = bias
    

    def mini_batch_weights(self):
        """
        Trains the logistic regression model using Mini-batch Gradient Descent.
        """
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
                    w_grad = self.weights_gradient(X, X_data[f"{house}_label"], h)
                    b_grad = self.bias_gradient(X_data[f"{house}_label"], h)
                    weight = weight - self.lr * w_grad
                    bias = bias - self.lr * b_grad
            self.weights[house] = weight.tolist()
            self.bias[house] = bias 
    

    def calculate_sgd(self):
        """
        Performs Stochastic Gradient Descent (SGD) to optimize logistic regression weights.
        """
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
        """
        Converts the learned weights and bias from standardized form back to original scale.
        """
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
        """
        Exports the trained model's weights and bias to a JSON file.
        """
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
