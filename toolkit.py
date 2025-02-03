import pandas as pd

class DataParser:
    @staticmethod
    def open_file(file_path, index_col="Index"):
        try:
            data = pd.read_csv(file_path, index_col=index_col)
            return data
        except Exception as e:
            print()
            exit(1)


    @staticmethod
    def replace_nan_values(data):
        for col in data.columns:
            if data[col].isnull().any():
                non_nan_values = data[col][data[col].notnull()]
                if not(non_nan_values.empty):
                    column_mean = sum(non_nan_values) / len(non_nan_values)
                    data[col] = data[col].fillna(column_mean)
        return data