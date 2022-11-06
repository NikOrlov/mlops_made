import pandas as pd
from tests.fake_data import FakeDatasetBuilder

DATA_PATH = '../data/raw/heart_cleveland_upload.csv'
TARGET_COLUMN = 'condition'

df = pd.read_csv(DATA_PATH)
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
fake_dataset_builder = FakeDatasetBuilder(df, categorical_columns, numerical_columns, TARGET_COLUMN)
