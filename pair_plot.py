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
        self.data =  DataParser.replace_nan_values(data)
        houses = self.data[house_col].unique()        
        features = self.data.drop(columns=['Hogwarts House']).columns
        rows, cols = 3, 5
        df = sns.pairplot(features, )
        plt.show()
        plt.close()


def main():
    hg = PairPlot()
    hg.pair_plot('datasets/dataset_train.csv')


if __name__ == "__main__":
    main()

