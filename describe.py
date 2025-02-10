import sys
from toolkit import DataParser

class DataAnalysis:
    '''Displays information for all numerical features of a dataset.'''

    def __init__(self, dataset):
        '''
        Inits the class, open the file and clean the data of non-numeric features and NaN values.

        Parameters:
        dataset (str): Path to dataset
        '''
        try:
            self.data_dict = DataParser.open_file(dataset)
            self.num_data, self.nan_data = DataParser.clean_data(self.data_dict)
            self.print_calc()
        except Exception as e:
            print(e)
            exit(1)

    def count(self, col_num):
        '''
        Calculates the length of a column.

        Parameters:
        col_num (pd.Series.column): A numeric column of a dataset

        Returns:
        int: Length of the column
        '''
        return len(col_num)
    
    def mean(self, col_num):
        '''
        Calculates the mean of a column.

        Parameters:
        col_num (pd.Series.column): A numeric column of a dataset

        Returns:
        float: Mean of the column
        '''
        total = 0.0
        for num in col_num:
            total += num
        return (total / len(col_num))
    
    def min(self, col_num):
        '''
        Calculates the minimum value of a column.

        Parameters:
        col_num (pd.Series.column): A numeric column of a dataset

        Returns:
        float: Minimum value of the column
        '''
        min_num = col_num[0]
        for num in col_num:
            if num < min_num:
                min_num = num
        return min_num
    
    def max(self, col_num):
        '''
        Calculates the maximum value of a column.

        Parameters:
        col_num (pd.Series.column): A numeric column of a dataset

        Returns:
        float: Maximum value of the column
        '''
        max_num = col_num[0]
        for num in col_num:
            if num > max_num:
                max_num = num
        return max_num
    
    def std(self, col_num):
        '''
        Calculates the standard deviation of a column.

        Parameters:
        col_num (pd.Series.column): A numeric column of a dataset

        Returns:
        float: Standard deviation of the column
        '''
        mean = self.mean(col_num)
        std = 0.0
        for num in col_num:
            std += (num - mean) ** 2
        return ((std / len(col_num)) ** 0.5)
    
    def range(self, col_num):
        '''
        Calculates the range of values in a column.

        Parameters:
        col_num (pd.Series.column): A numeric column of a dataset

        Returns:
        float: Range of values in a column
        '''
        return (self.max(col_num) - self.min(col_num))

    def nan_count(seld, col_num):
        '''
        Calculates the number of NaN in a column.

        Parameters:
        col_num (pd.Series.column): A numeric column of a dataset

        Returns:
        float: Number of NaN in a column
        '''
        count = 0
        for num in col_num:
            if num != num:
                count +=1
        return count
    
    def quicksort(self, col_num):
        '''
        Sorts the values from lowest to highest of a column with quicksort algorithm.

        Parameters:
        col_num (pd.Series.column): A numeric column of a dataset

        Returns:
        pd.Series.column: The column with the numbers sorted
        '''
        if len(col_num) <= 1:
            return col_num
        else:
            pivot = col_num[0]
            lower = [x for x in col_num[1:] if x <= pivot]
            higher = [x for x in col_num[1:] if x > pivot]
            return self.quicksort(lower) + [pivot] + self.quicksort(higher)

    def quantile(self, col_num, percentage):
        '''
        Calculates the quantile of a column.

        Parameters:
        col_num (pd.Series.column): A numeric column of a dataset
        percentage (int): Percentage of the quantile to be calculated

        Returns:
        float: Quantile of a column
        '''
        sorted_num = self.quicksort(col_num)
        if (percentage == 25):
            return float(sorted_num[int(len(col_num) / 4)])
        elif (percentage == 50):
            return float(sorted_num[int(2 * len(col_num) / 4)])
        elif (percentage == 75):
            return float(sorted_num[int(3 * len(col_num) / 4)])
    
    def print_calc(self):
        '''
        Prints all the calculations of data analysis from a dataset.
        '''
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
            print(f"{func_name:<10}", end=' ')
            if (func_name == "Nan values"):
                for col in self.nan_data.columns:
                    print(f"{func(self.nan_data[col].to_list()):>13.6f}", end=' ')
            else:
                for col in self.num_data.columns:
                    print(f"{func(self.num_data[col].to_list()):>13.6f}", end=' ')
            print()

def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./describe.py dataset_name")
        sys.exit(1)

    data = DataAnalysis(sys.argv[1])

if __name__ == "__main__":
    main()
