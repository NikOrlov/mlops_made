from unittest import TestCase
import unittest
import os
import pickle
from predict import predict_pipeline
from src.make_dataset import read_config
from train import train_pipeline


with open('tests/fake_dataset_builder.pkl', 'rb') as file:
    fake_dataset_builder = pickle.load(file)


class TestPredict(TestCase):
    def test_missed_column(self):
        config_path = 'tests/log_reg_test.yaml'
        params = read_config(config_path)

        data = fake_dataset_builder.generate_dataset()
        data.to_csv(params.input_data_path)

        train_pipeline(params)

        model_path = 'tests/models/model_test.pkl'
        data_path = 'data_missed_column.csv'
        dataframe = fake_dataset_builder.generate_dataset()
        dataframe.drop('sex', axis=1, inplace=True)
        dataframe.to_csv(data_path)
        output_path = 'error'
        with self.assertRaises(KeyError):
            predict_pipeline(model_path, data_path, output_path)
        os.remove(params.input_data_path)
        os.remove(data_path)

    def test_renamed_column(self):
        config_path = 'tests/log_reg_test.yaml'
        params = read_config(config_path)

        data = fake_dataset_builder.generate_dataset()
        data.to_csv(params.input_data_path)

        train_pipeline(params)

        model_path = 'tests/models/model_test.pkl'
        data_path = 'data_renamed_column.csv'
        output_path = 'error'
        dataframe = fake_dataset_builder.generate_dataset()
        dataframe.rename(columns={'sex': 'sex123'}, inplace=True)
        dataframe.to_csv(data_path)

        with self.assertRaises(KeyError):
            predict_pipeline(model_path, data_path, output_path)

        os.remove(params.input_data_path)
        os.remove(data_path)
        os.remove(params.metric_path)
        os.remove(params.output_model_path)
        os.removedirs(os.path.dirname(params.output_model_path))
        os.removedirs(os.path.dirname(params.metric_path))


if __name__ == '__main__':
    unittest.main()
