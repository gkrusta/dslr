import matplotlib.pyplot as plt
from toolkit import DataParser


class Histogram():
    def __init__(self):
        self.data = None


    def histogram(self, infile, house_col = "Hogwarts House"):
        all_data = DataParser.open_file(infile)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand']
        data = all_data.drop(columns=columns_to_drop, errors='ignore')
        self.data =  DataParser.replace_nan_values(data)
        houses = self.data[house_col].unique()        
        features = self.data.drop(columns=['Hogwarts House'] ).columns
        rows, cols = 3, 5
        fig, axs = plt.subplots(rows, cols, figsize=(12,8))
        fig.suptitle("Histogram", fontsize=16, fontweight="bold")
        axs = axs.flatten()

        for i, feature in enumerate(features):
            for house in houses:
                house_data = self.data[self.data[house_col] == house]
                axs[i].hist(house_data[feature], bins=20, alpha = 0.7, label=house)
            axs[i].set_title(feature, fontsize=10, fontweight="bold", pad=2)
            axs[i].tick_params(axis='both', labelsize=6)

        h, l = axs[0].get_legend_handles_labels()
        for i in range(len(features), len(axs)):
            axs[i].set_visible(False)

        fig.text(0.5, 0.04, 'Values', ha='center', fontweight="bold")
        fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontweight="bold")
        fig.legend(h, l, loc='lower right', fontsize=12, bbox_to_anchor=(0.85, 0.15))
        plt.show()
        plt.close()


def main():
    hg = Histogram()
    hg.histogram('datasets/dataset_train.csv')


if __name__ == "__main__":
    main()
