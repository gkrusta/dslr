import matplotlib.pyplot as plt
from toolkit import DataParser

class Histogram():
    def __init__(self):
        self.data = None


    def visualize(self, feature):
        """
        Visualize the data distribution and optionally the regression line.
        """
        try:
            plt.xlabel("Values", fontsize=14, fontweight="bold", labelpad=20)
            plt.ylabel('Frequency', fontsize=14, fontweight="bold", labelpad=20)
            plt.legend(loc="upper right", fontsize=10, title='Hogwarts House')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title(feature, fontsize=20, fontweight="bold", pad=20)
            plt.grid(True)
            plt.show()
            plt.close()
        except KeyboardInterrupt:
            print("\nVisualization interrupted by the user. Exiting cleanly.")
        except Exception as e:
            print(f"Error: {e}")


    def histogram(self, infile, house_col = "Hogwarts House"):
        all_data = DataParser.open_file(infile)
        columns_to_drop = ['First Name', 'Last Name', 'Birthday', 'Best Hand']
        data = all_data.drop(columns=columns_to_drop, errors='ignore')
        self.data =  DataParser.replace_nan_values(data)
        houses = self.data[house_col].unique()
        #self.data.reset_index(inplace=True)
        
        features = self.data.drop(columns=['Hogwarts House'] ).columns
        print("features ", features)
        rows, cols = 3, 5
        fig, axs = plt.subplots(rows, cols, figsize=(12,8))
        axs = axs.flatten()
        for i, feature in enumerate(features):
            for house in houses:
                house_data = self.data[self.data[house_col] == house]
                #print("house_data", house_data)
                axs[i].hist(house_data[feature], bins=20, alpha = 0.7, label=house)
            axs[i].set_title(feature, fontsize=10, fontweight="bold", pad=2)
            #axs[i].set_xlabel("Values", fontsize=6, fontweight="bold", labelpad=1)
            #axs[i].set_ylabel("Frequency", fontsize=6, fontweight="bold", labelpad=1)
            #self.visualize(feature)
        h, l = axs[0].get_legend_handles_labels()
        #plt.legend(h, l, loc='center')
        fig.text(0.5, 0.04, 'common X', ha='center', fontweight="bold")
        fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical', fontweight="bold")
        fig.legend(h, l, loc='upper right', bbox_to_anchor=(1.01, 1))
        plt.show()


def main():
    hg = Histogram()
    hg.histogram('datasets/dataset_train.csv')


if __name__ == "__main__":
    main()
