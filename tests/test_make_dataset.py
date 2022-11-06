from unittest import TestCase
import unittest
import dataclasses
import pandas as pd
from src.make_dataset import split_data, read_config
from src.entities import SplittingParams, FeatureParams, TrainingParams, TrainingPipelineParams
from tests.data_generator import fake_dataset_builder


class TestReadData(TestCase):
    def test_split_data(self):
        num_rows = 100
        params = SplittingParams(test_size=0.3, random_state=42, shuffle=False)
        test_size = int(num_rows * params.test_size)
        dataframe = fake_dataset_builder.generate_dataset(num_rows)
        df_train_true = dataframe[:-test_size]
        df_test_true = dataframe[-test_size:]

        df_train_my, df_test_my = split_data(dataframe, params)
        self.assertTrue(pd.DataFrame.equals(df_train_my, df_train_true))
        self.assertTrue(pd.DataFrame.equals(df_test_my, df_test_true))

    def test_read_config(self):
        input_path = 'data/raw/heart_cleveland_upload.csv'
        output_path = 'models/model.pkl'
        metric_path = 'metrics/metric.json'
        splitting_params = SplittingParams(test_size=0.3, random_state=42, shuffle=True)
        feature_params = \
            FeatureParams(numerical_columns=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                          categorical_columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
                          columns_to_drop=None,
                          target_column='condition',
                          fill_na_numerical_strategy='mean',
                          fill_na_categorical_strategy='most_frequent')
        training_params = TrainingParams(model='LogisticRegression',
                                         model_params={
                                             'penalty': 'none'
                                         },
                                         random_state=42)

        params_true = TrainingPipelineParams(input_data_path=input_path,
                                             output_model_path=output_path,
                                             metric_path=metric_path,
                                             splitting_params=splitting_params,
                                             feature_params=feature_params,
                                             training_params=training_params)

        config_path = 'configs/log_reg_no_regularization.yaml'
        param_my = read_config(config_path)
        self.assertDictEqual(dataclasses.asdict(param_my), dataclasses.asdict(params_true))


if __name__ == '__main__':
    unittest.main()
