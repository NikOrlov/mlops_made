import pandas as pd
import numpy as np


class Feature:
    def __init__(self, name, is_categorical, min_val, max_val):
        self.name = name
        self.is_categorical = is_categorical
        self.min_val = min_val
        self.max_val = max_val
        self.data = None

    def generate_data(self, num_rows):
        if self.is_categorical:
            self.data = np.random.randint(self.min_val, self.max_val + 1, num_rows)
        else:
            self.data = np.random.uniform(self.min_val, self.max_val, num_rows).astype(int)


class FakeDatasetBuilder:
    def __init__(self, df: pd.DataFrame, categorical_columns, numerical_columns, target):
        self.df = df
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target = target
        self.features_info = {}

    def build_features_info(self):
        for column in self.df.columns:
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if column == self.target or column in self.categorical_columns:
                is_categorical = True
            else:
                is_categorical = False
            self.features_info[column] = Feature(column, is_categorical, min_val, max_val)

    def generate_dataset(self, num_rows=100) -> pd.DataFrame:
        if not self.features_info:
            self.build_features_info()

        df_data = {}
        for column, feature in self.features_info.items():
            feature.generate_data(num_rows)
            df_data[column] = feature.data
        df = pd.DataFrame.from_dict(df_data)
        return df


if __name__ == '__main__':
    data_path = '../data/raw/heart_cleveland_upload.csv'
    df = pd.read_csv(data_path)
    numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    target_column = 'condition'
    fake_dataset_builder = FakeDatasetBuilder(df, categorical_columns, numerical_columns, target_column)
    fake_dataset = fake_dataset_builder.generate_dataset(10)
