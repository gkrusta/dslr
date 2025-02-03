import sys
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
    
    def range(self, col_num):
        return (self.max(col_num) - self.min(col_num))

    def nan_count(seld, col_num):
        count = 0
        for num in col_num:
            if num != num:
                count +=1
        return count
    
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

    def read_file(self, dataset):
        try:
            self.data_dict = DataParser.open_file(dataset)
            columns_name = []
            for name in self.data_dict.columns:
                if isinstance(self.data_dict[name].iloc[0], int) or isinstance(self.data_dict[name].iloc[0], float):
                    columns_name.append(name)

            self.num_data = self.data_dict.loc[:, columns_name]
            self.data_dict = self.num_data.copy()
        except:
            print("Coludn't read the dataset")
            sys.exit(1)
        self.num_data = DataParser.replace_nan_values(self.num_data)
    
    def print_calc(self):
        math_func = {
            "Count": self.count,
            "Mean": self.mean,
            "Std": self.std,
            "Min": self.min,
            "25%": lambda col: self.quantile(col, 25),
            "50%": lambda col: self.quantile(col, 50),
            "75%": lambda col: self.quantile(col, 75),
            "Max": self.max,
            "Range": self.range,
            "Nan values": self.nan_count
        }

        print(f"{'':<8}", end=' ')
        for col in self.num_data.columns:
            print(f"{col:>5}", end=' ')
        print()

        for func_name, func in math_func.items():
            print(f"{func_name:<7}", end=' ')
            if (func_name == "Nan values"):
                for col in self.data_dict.columns:
                    print(f"{func(self.data_dict[col].to_list()):>13.6f}", end=' ')
            else:
                for col in self.num_data.columns:
                    print(f"{func(self.num_data[col].to_list()):>13.6f}", end=' ')
            print()

def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./describe.py dataset_name")
        sys.exit(1)

    data = DataAnalysis()
    data.read_file(sys.argv[1])
    data.print_calc()

if __name__ == "__main__":
    main()
