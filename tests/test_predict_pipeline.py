from unittest import TestCase
import unittest
import os
from predict import predict_pipeline
from tests.data_generator import fake_dataset_builder


class TestPredict(TestCase):
    def test_missed_column(self):
        model_path = 'models/model.pkl'
        data_path = 'data_missed_column.csv'
        dataframe = fake_dataset_builder.generate_dataset()
        dataframe.drop('sex', axis=1, inplace=True)
        dataframe.to_csv(data_path)
        output_path = 'error'
        with self.assertRaises(KeyError):
            predict_pipeline(model_path, data_path, output_path)
        os.remove(data_path)

    def test_renamed_column(self):
        model_path = 'models/model.pkl'
        data_path = 'data_renamed_column.csv'
        output_path = 'error'
        dataframe = fake_dataset_builder.generate_dataset()
        dataframe.rename(columns={'sex': 'sex123'}, inplace=True)
        dataframe.to_csv(data_path)

        with self.assertRaises(KeyError):
            predict_pipeline(model_path, data_path, output_path)
        os.remove(data_path)


if __name__ == '__main__':
    unittest.main()
