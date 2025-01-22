import pandas as pd
import numpy as np
from utils import replace_nan_values

class Logisticregression():
    def __init__(self):
        self.lr = 0.001
        self.iterations = 1000
        self.weights = 0
        self.bias = 0
        self.data = None


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_cost(self, y_label, y_predicted):
        epsilon = 1e-9
        


    def parse_arguments(self):
        all_data = pd.read_csv('datasets/dataset_train.csv')
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Defense Against the Dark Arts'])
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        labels = {}
        data = replace_nan_values(data, columns_to_drop)
        for house in houses:
            labels[house] = (data['Hogwarts House'] == house).astype(int)
        
        for house, label in labels.items():
            data[f'{house}_label'] = label

        print(data.head())



def main():
    lr = Logisticregression()
    lr.data = Logisticregression.parse_arguments()



if __name__ == "__main__":
    main()