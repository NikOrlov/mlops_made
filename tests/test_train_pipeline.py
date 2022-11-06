from unittest import TestCase
import unittest
import os
from make_dataset import read_config
from train import train_pipeline


class TestCaseBase(TestCase):
    def assertIsFile(self, path):
        if not os.path.isfile(path):
            raise AssertionError('File does not exist: %s' % str(path))


class TestTrain(TestCaseBase):
    def test_train_pipeline(self):
        config_path = 'log_reg_test.yaml'
        params = read_config(config_path)
        train_pipeline(params)
        self.assertIsFile(params.output_model_path)
        self.assertIsFile(params.metric_path)

        os.remove(params.metric_path)
        os.remove(params.output_model_path)
        os.removedirs(os.path.dirname(params.output_model_path))
        os.removedirs(os.path.dirname(params.metric_path))


if __name__ == '__main__':
    unittest.main()
