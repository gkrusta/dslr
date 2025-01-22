

class DataParser:
    @staticmethod
    def replace_nan_values(data, drop_cols):
        data = data.drop(columns=drop_cols)
        for col in data.columns:
            if data[col].isnull().any():
                non_nan_values = data[col][data[col].notnull()]
                column_mean = sum(non_nan_values) / len(non_nan_values)
                data[col].fillna(column_mean, inplace=True)
        return data