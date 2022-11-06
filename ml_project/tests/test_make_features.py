from unittest import TestCase
import unittest
import numpy as np
import pandas as pd
from src.make_features import build_transformer, transform_data
from src.entities import FeatureParams


class TestMakeDataset(TestCase):
    def test_one_hot_encoding(self):
        params = FeatureParams(numerical_columns=[],
                               categorical_columns=['type'],
                               fill_na_categorical_strategy='most_frequent',
                               fill_na_numerical_strategy='mean',
                               columns_to_drop=[],
                               target_column='target')

        d_before = {'type': [1, 2, 3],
                    'target': [1, 0, 1]}
        df_before = pd.DataFrame(data=d_before)
        data_after = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        transformer = build_transformer(params)
        data_my = transform_data(df_before, transformer)
        self.assertListEqual(data_after.tolist(), data_my.tolist())

    def test_scaler(self):
        params = FeatureParams(numerical_columns=['age'],
                               categorical_columns=[],
                               fill_na_categorical_strategy='most_frequent',
                               fill_na_numerical_strategy='mean',
                               columns_to_drop=[],
                               target_column='target')

        d_before = {'age': [1, 2, 3],
                    'target': [1, 0, 1]}
        df_before = pd.DataFrame(data=d_before)

        data_after = df_before['age'].values
        data_after = (data_after - data_after.mean()) / data_after.std()
        data_after = data_after.reshape(3, 1)
        transformer = build_transformer(params)
        data_my = transform_data(df_before, transformer)
        self.assertListEqual(data_after.tolist(), data_my.tolist())

    def test_imp(self):
        params = FeatureParams(numerical_columns=['age'],
                               categorical_columns=[],
                               fill_na_categorical_strategy='most_frequent',
                               fill_na_numerical_strategy='mean',
                               columns_to_drop=[],
                               target_column='target')

        d_before = {'age': [None, 2, 3],
                    'target': [1, 0, 1]}
        df_before = pd.DataFrame(data=d_before)

        data_after = np.array([2.5, 2, 3])
        data_after = (data_after - data_after.mean()) / data_after.std()
        data_after = data_after.reshape(3, 1)
        transformer = build_transformer(params)
        data_my = transform_data(df_before, transformer)
        self.assertListEqual(data_after.tolist(), data_my.tolist())

    def test_all(self):
        params = FeatureParams(numerical_columns=['age'],
                               categorical_columns=['type'],
                               fill_na_categorical_strategy='most_frequent',
                               fill_na_numerical_strategy='mean',
                               columns_to_drop=[],
                               target_column='target')

        d_before = {'age': [None, 2, 3],
                    'type': [1, 2, 3],
                    'target': [1, 0, 1]}
        df_before = pd.DataFrame(data=d_before)

        data_after = np.array([2.5, 2, 3])
        data_after = (data_after - data_after.mean()) / data_after.std()
        data_after = data_after.reshape(3, 1)
        data_after = np.hstack((np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), data_after))
        transformer = build_transformer(params)
        data_my = transform_data(df_before, transformer)
        self.assertListEqual(data_after.tolist(), data_my.tolist())


if __name__ == '__main__':
    unittest.main()
