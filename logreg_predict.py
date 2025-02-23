import sys
from numpy import np
from toolkit import DataParser
import json

class Prediction():
    def __init__(self):
        self.data = None
        self.weights = None
    
    def predictions_file(self):
        try:
            with open("dataset_test.csv", "w") as csvfile:
                print("Index,Hogwarts House")
        except Exception as e:
            print(e)
            sys.exit(1)

    def predict_house(X):
        probabilities = {}

        for 

    def parse_arguments(self, dataset, weights):
        all_data = DataParser.open_file(dataset)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Defense Against the Dark Arts']
        all_data = all_data.drop(columns=columns_to_drop, errors='ignore')
        self.data =  DataParser.replace_nan_values(all_data)
        with open(weights, "r") as file:
            self.weights = json.load(file)
        for index, row in self.data.iterrows():
            X = np.array(row)
            predicted_house = predict_house(X)

def main():
    if (len(sys.argv) < 3):
        print("Usage: python3 ./logreg_predict.py dataset_name weights.json")
        sys.exit(1)
    pr = Prediction()
    pr.parse_arguments(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()