import pandas as pd
import numpy as np


class Logisticregression():
    def __init__(self):
        self.lr = 0.001
        self.iterations = 1000
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    all_data = pd.read_csv('datasets/dataset_train.csv')
    data = all_data.drop(columns=['First Name', 'Last Name', 'Birthday', 'Best Hand'])
    
    print(data.head())


def main()


if __name__ == "__main__":
    main()