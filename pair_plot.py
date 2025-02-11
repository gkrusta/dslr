import seaborn as sns
import matplotlib.pyplot as plt
from toolkit import DataParser


class PairPlot():
    def __init__(self):
        self.data = None


    def pair_plot(self, infile, house_col = "Hogwarts House"):
        """
        Creates a pair plot (scatter plot matrix) to visualize the relationships between numerical features.
        This function reads the dataset, preprocesses the data, and plots each feature 
        against every other feature in a grid format.
        The diagonal plots display histograms of individual features,
        while the off-diagonal plots show scatter plots to analyze 
        potential correlations.
        """
        all_data = DataParser.open_file(infile)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand']
        data = all_data.drop(columns=columns_to_drop, errors='ignore')
        data = data.rename(columns={'Defense Against the Dark Arts': 'DADA'})
        self.data =  DataParser.replace_nan_values(data)
        sns.pairplot(self.data, hue=house_col)
        plt.show()
        plt.close()


def main():
    hg = PairPlot()
    hg.pair_plot('datasets/dataset_train.csv')


if __name__ == "__main__":
    main()
