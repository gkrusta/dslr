import matplotlib.pyplot as plt
from toolkit import DataParser

class ScatterPlot:
    def __init__(self):
        self.data = []
        self.num_data = []
    
    def similar_features(self, dataset):
        self.data = DataParser.open_file(dataset)
        columns_name = []
        for name in self.data.columns:
            if isinstance(self.data[name].iloc[0], int) or isinstance(self.data[name].iloc[0], float):
                columns_name.append(name)

        self.num_data = self.data.loc[:, columns_name]
        self.num_data = DataParser.replace_nan_values(self.num_data)

        classes = ["Ravenclaw","Slytherin","Gryffindor","Hufflepuff"]
        for house in classes:
            class_data = self.num_data[self.data["Hogwarts House"] == house]
            plt.scatter(class_data["Astronomy"], class_data["Defense Against the Dark Arts"], label=house)

        plt.xlabel("Astronomy")
        plt.ylabel("Defense Against the Dark Arts")
        plt.title("Similar Features")
        plt.legend()
        plt.show()

def main():
    sp = ScatterPlot()
    sp.similar_features("datasets/dataset_train.csv")

if __name__ == "__main__":
    main()