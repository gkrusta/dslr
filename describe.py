import sys
import pandas as pd
from toolkit import DataParser

class DataAnalysis:

    def __init__(self):
        self.data_dict = []
        self.num_data = []

    def count(self, col_num):
        return len(col_num)
    
    def mean(self, col_num):
        total = 0.0
        for num in col_num:
            total += num
        return (total / len(col_num))
    
    def min(self, col_num):
        min_num = col_num[0]
        for num in col_num:
            if num < min_num:
                min_num = num
        return min_num
    
    def max(self, col_num):
        max_num = col_num[0]
        for num in col_num:
            if num > max_num:
                max_num = num
        return max_num
    
    def std(self, col_num):
        mean = self.mean(col_num)
        std = 0.0
        for num in col_num:
            std += (num - mean) ** 2
        return ((std / len(col_num)) ** 0.5)
    
    def quicksort(self, col_num):
        if len(col_num) <= 1:
            return col_num
        else:
            pivot = col_num[0]
            lower = [x for x in col_num[1:] if x <= pivot]
            higher = [x for x in col_num[1:] if x > pivot]
            return self.quicksort(lower) + [pivot] + self.quicksort(higher)

    def quantile(self, col_num, percentage):
        sorted_num = self.quicksort(col_num)

        if (percentage == 25):
            return float(sorted_num[int(len(col_num) / 4)])
        elif (percentage == 50):
            return float(sorted_num[int(2 * len(col_num) / 4)])
        elif (percentage == 75):
            return float(sorted_num[int(3 * len(col_num) / 4)])

    def nan_values(self, col_num, mean):
        for num in col_num:
            if num != num:
                num = mean

    def read_file(self, dataset):
        try:
            self.data_dict = DataParser.open_file(dataset)
            columns_name = []
            for name in self.data_dict.columns.values:
                if isinstance(self.data_dict[name].iloc[0], int) or isinstance(self.data_dict[name].iloc[0], float):
                    columns_name.append(name)

            self.num_data = self.data_dict.loc[:, columns_name]
        except:
            print("Coludn't read the dataset")
            sys.exit(1)
        self.num_data = DataParser.replace_nan_values(self.num_data)

    
    def print_calc(self):
        math_func = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", " ", " ", " ", " ", " "]
        for i in range(len(self.num_data.columns)):
            print(self.num_data.columns.values[i], end=' ')
        for i in range(len(self.num_data.columns)):
            print('{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f} {6:.6f} {7:.6f} {8:.6f}'.format(
                math_func[i],
                self.count(self.num_data.iloc[:,i].to_list()),
                self.mean(self.num_data.iloc[:,i].to_list()),
                self.std(self.num_data.iloc[:,i].to_list()),
                self.min(self.num_data.iloc[:,i].to_list()),
                self.quantile(self.num_data.iloc[:,i].to_list(), 25),
                self.quantile(self.num_data.iloc[:,i].to_list(), 50),
                self.quantile(self.num_data.iloc[:,i].to_list(), 75),
                self.max(self.num_data.iloc[:,i].to_list())
            ))


def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./describe.py dataset_name")
        sys.exit(1)

    data = DataAnalysis()
    data.read_file(sys.argv[1])
    data.print_calc()

if __name__ == "__main__":
    main()