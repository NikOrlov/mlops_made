from unittest import TestCase
import unittest
import os
import pickle
from src.make_dataset import read_config
from train import train_pipeline


with open('tests/fake_dataset_builder.pkl', 'rb') as file:
    fake_dataset_builder = pickle.load(file)


class TestCaseBase(TestCase):
    def assertIsFile(self, path):
        if not os.path.isfile(path):
            raise AssertionError('File does not exist: %s' % str(path))


class TestTrain(TestCaseBase):
    def test_train_pipeline(self):
        config_path = 'tests/log_reg_test.yaml'
        params = read_config(config_path)

        data = fake_dataset_builder.generate_dataset()
        data.to_csv(params.input_data_path)

        train_pipeline(params)
        self.assertIsFile(params.output_model_path)
        self.assertIsFile(params.metric_path)

        os.remove(params.input_data_path)
        os.remove(params.metric_path)
        os.remove(params.output_model_path)
        os.removedirs(os.path.dirname(params.output_model_path))
        os.removedirs(os.path.dirname(params.metric_path))


if __name__ == '__main__':
    unittest.main()
