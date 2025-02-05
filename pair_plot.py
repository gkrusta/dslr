import seaborn as sns
import matplotlib.pyplot as plt
from toolkit import DataParser


class PairPlot():
    def __init__(self):
        self.data = None


    def pair_plot(self, infile, house_col = "Hogwarts House"):
        all_data = DataParser.open_file(infile)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand']
        data = all_data.drop(columns=columns_to_drop, errors='ignore')
        data = data.rename(columns={'Defense Against the Dark Arts': 'DADA'})
        self.data =  DataParser.replace_nan_values(data)
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        #sns.pairplot(self.data, hue=house_col)
        num_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        print(num_features)
        fig, axs = plt.subplots(len(num_features), len(num_features), figsize=(30, 30))
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
                        axs[i, j].scatter(house_data[num_features[i]], house_data[num_features[j]], label=house)
                if j != 0:
                    axs[i, j].set_yticklabels([])
                if i != len(num_features) - 1:
                    axs[i, j].set_xticklabels([])

        # Add a single legend outside the plot
        handles = [plt.Line2D([0], [0], marker='o', color='w')]
        fig.legend(handles, houses, loc='upper right')
        plt.tight_layout()
        plt.show()
        plt.close()


def main():
    hg = PairPlot()
    hg.pair_plot('datasets/dataset_train.csv')


if __name__ == "__main__":
    main()
