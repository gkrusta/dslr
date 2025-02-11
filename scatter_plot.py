import sys
import matplotlib.pyplot as plt
from toolkit import DataParser

class ScatterPlot:
    '''Creates a scatter plot of the two features with the highest correlation.'''

    def __init__(self, dataset):
        '''
        Inits the class, open and clean the dataset and calls similar_features function.
        Parameters:
        dataset (str): Path to dataset
        '''
        self.data = DataParser.open_file(dataset)
        self.num_data, _ = DataParser.clean_data(self.data)
        self.similar_features()
    
    def similar_features(self):
        '''
        Calculates the correlation of all features. Finds the highest correlations and show it in a scatter plot.
        '''
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
    if (len(sys.argv) != 1):
        print("Usage: python3 ./scatter_plot.py")
        sys.exit(1)

    sp = ScatterPlot("datasets/dataset_train.csv")

if __name__ == "__main__":
    main()