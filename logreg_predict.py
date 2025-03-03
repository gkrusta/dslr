import sys
import numpy as np
import pandas as pd
from pyrsistent import optional
from toolkit import DataParser
import json


class Prediction():
    def __init__(self):
        self.data = None
        self.weights = None


    def get_probability(self, X):
        probabilities = {}

        for house, params in self.weights.items():
            W = np.array(params['weights'])
            b = params['bias']
            probabilities[house] = DataParser.sigmoid(np.dot(X, W) + b)
        #formated_probs = {house: f"{prob:.12f}" for house, prob in probabilities.items()}
        #print(formated_probs)
        return max(probabilities, key=probabilities.get)


    def predict_house(self):
        predictions = []

        for index, row in self.data.iterrows():
            X = np.array(row)
            predicted_house = Prediction.get_probability(self, X)
            #print("predicted_house ", predicted_house)
            predictions.append([index, predicted_house])
        df = pd.DataFrame(predictions, columns=['index','Hogwarts House'])
        df.to_csv('houses.csv', index=False)


    def parse_arguments(self, dataset, weights):
        all_data = DataParser.open_file(dataset)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Defense Against the Dark Arts', 'Hogwarts House']
        all_data = all_data.drop(columns=columns_to_drop, errors='ignore')
        self.data =  DataParser.replace_nan_values(all_data)
        with open(weights, "r") as file:
            self.weights = json.load(file)


def predict(test_path, weights_path, output_folder=optional, config_path=optional):
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
