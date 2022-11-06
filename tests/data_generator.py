import pandas as pd
from tests.fake_data import FakeDatasetBuilder

data_path = '../data/raw/heart_cleveland_upload.csv'
df = pd.read_csv(data_path)
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
target_column = 'condition'
fake_dataset_builder = FakeDatasetBuilder(df, categorical_columns, numerical_columns, target_column)
