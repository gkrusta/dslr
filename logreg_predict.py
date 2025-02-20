import sys
from toolkit import DataParser

class Prediction():
    def __init__(self):
        self.data = None
    
    def predictions_file(self):
        try:
            with open("dataset_test.csv", "w") as csvfile:
                print("Index,Hogwarts House")
        except Exception as e:
            print(e)
            sys.exit(1)

    def parse_arguments(self, dataset, weights):
        all_data = DataParser.open_file(dataset)
        self.data =  DataParser.replace_nan_values(all_data)

def main():
    if (len(sys.argv) < 3):
        print("Usage: python3 ./logreg_predict.py dataset_name weights.json")
        sys.exit(1)
    pr = Prediction()
    pr.parse_arguments(sys.argv[1], sys.argv[2])



if __name__ == "__main__":
    main()