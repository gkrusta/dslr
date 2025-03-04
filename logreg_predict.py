import sys
import numpy as np
import pandas as pd
from pyrsistent import optional
from toolkit import DataParser
import json


class Prediction():
    """
    Handles the prediction of Hogwarts houses using a trained logistic regression model.

    This class loads a dataset and a set of trained weights, applies the logistic regression
    model to predict the most likely house for each example, and saves the results to a CSV file.
    """
    def __init__(self):
        """
        Inits the class and the variables.
        """
        self.data = None
        self.weights = None


    def get_probability(self, X):
        """
        Computes the probability of a sample belonging to each Hogwarts house using the sigmoid function 
        and selects the house with the highest probability.
        """
        probabilities = {}

        for house, params in self.weights.items():
            W = np.array(params['weights'])
            b = params['bias']
            probabilities[house] = DataParser.sigmoid(np.dot(X, W) + b)
        return max(probabilities, key=probabilities.get)


    def predict_house(self):
        """
        Predicts the Hogwarts house for each sample in the dataset using trained weights and saves 
        the results to 'houses.csv'.
        """
        predictions = []

        for index, row in self.data.iterrows():
            X = np.array(row)
            predicted_house = Prediction.get_probability(self, X)
            predictions.append([index, predicted_house])
        df = pd.DataFrame(predictions, columns=['index','Hogwarts House'])
        df.to_csv('houses.csv', index=False)


    def parse_arguments(self, dataset, weights):
        """
        Loads the dataset, removes unnecessary columns, replaces NaN values,
        and loads pre-trained weights from a file.
        """
        all_data = DataParser.open_file(dataset)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Defense Against the Dark Arts', 'Hogwarts House']
        all_data = all_data.drop(columns=columns_to_drop, errors='ignore')
        self.data =  DataParser.replace_nan_values(all_data)
        with open(weights, "r") as file:
            self.weights = json.load(file)


def predict(test_path, weights_path, output_folder=optional, config_path=optional):
    """
    Loads test data and trained weights, performs predictions, and saves the results.
    """
    pr = Prediction()
    pr.parse_arguments(test_path, weights_path,)
    pr.predict_house()


def main():
    if (len(sys.argv) < 3): 
        print("Usage: python3 ./logreg_predict.py dataset_name weights.json")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
