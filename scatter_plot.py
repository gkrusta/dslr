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

        corr_data = self.num_data.corr()
        max_corr = 0
        feature_corr = []
        for feature in self.num_data.columns:
            for feature_2 in self.num_data.columns:
                if feature == feature_2:
                    continue
                else:
                    corr_value = abs(corr_data[feature][feature_2])
                    if corr_value > max_corr:
                        max_corr = corr_value
                        feature_corr = [feature, feature_2]

        classes = self.data["Hogwarts House"].unique()
        for house in classes:
            class_data = self.num_data[self.data["Hogwarts House"] == house]
            plt.scatter(class_data[feature_corr[0]], class_data[feature_corr[1]], label=house)

        plt.xlabel(feature_corr[0])
        plt.ylabel(feature_corr[1])
        plt.title("Similar Features")
        plt.legend()
        plt.show()

def main():
    sp = ScatterPlot()
    sp.similar_features("datasets/dataset_train.csv")

if __name__ == "__main__":
    main()