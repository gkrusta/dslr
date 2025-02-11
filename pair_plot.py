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
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        #sns.pairplot(self.data, hue=house_col)
        num_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        print(num_features)
        fig, axs = plt.subplots(len(num_features), len(num_features), figsize=(30, 34))
        fig.suptitle("Pair plot", fontsize=16, fontweight="bold")

        for i in range(len(num_features)):
            for j in range(len(num_features)):
                if i == j:
                    for house in houses:
                        house_data = self.data[self.data[house_col] == house]
                        axs[i, j].hist(house_data[num_features[i]], bins=20, alpha = 0.7, label=house)
                else:
                    for house in houses:
                        house_data = self.data[self.data[house_col] == house]
                        axs[i, j].scatter(house_data[num_features[i]], house_data[num_features[j]], label=house, s=5)
                if j == 0:
                    axs[i, j].set_ylabel(num_features[i], fontsize=8, fontweight="bold")
                else:
                    axs[i, j].set_yticklabels([])
                if i != len(num_features) - 1:
                    axs[i, j].set_xticklabels([])
                elif i == len(num_features) - 1:
                    axs[i, j].set_xlabel(num_features[j], fontsize=8, fontweight="bold")

        h, l = axs[0,0].get_legend_handles_labels()
        fig.legend(h, l, loc='upper right', bbox_to_anchor=(0.98, 0.995), fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 0.99, 0.95])
        plt.show()
        plt.close()


def main():
    hg = PairPlot()
    hg.pair_plot('datasets/dataset_train.csv')


if __name__ == "__main__":
    main()
