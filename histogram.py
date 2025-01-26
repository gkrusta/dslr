import matplotlib.pyplot as plt
import numpy as np
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
            plt.legend(loc="upper right", fontsize=10)
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
        data = all_data.drop(columns=columns_to_drop)
        self.data =  DataParser.replace_nan_values(data)
        houses = self.data[house_col].unique()
        #self.data.reset_index(inplace=True)
        
        features = self.data.drop(columns=['Hogwarts House'] ).columns
        print("features ", features)

        for feature in features:
            plt.figure(figsize=(10, 8))
            for house in houses:
                house_data = self.data[house_col] == house
                plt.hist(house_data[feature], bins=20, alpha = 0.7)
            self.visulize(feature) 


def main():
    hg = Histogram()
    hg.histogram('datasets/dataset_train.csv')


if __name__ == "__main__":
    main()
