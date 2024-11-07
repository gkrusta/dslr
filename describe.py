import sys
import pandas as pd

class DataAnalysis:
    def __init__():
        self.data_dict = []

    def read_file():
        try:
            dataset = pd.read_csv("dataset_train.csv")
            self.data_dict = dataset.to_dict(orient='list')
        except:
            print("Coludn't read the dataset")
            sys.exit(1)


def main():
    if (sys.argv < 2):
        print("Usage: python3 ./describe.py dataset_name")
        sys.exit(1)
    
    data = DataAnalysis()
    data.read_file()

if __name__ == "__main__":
    main()