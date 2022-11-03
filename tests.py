import unittest
from unittest import TestCase
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from faker import Faker
from entities import SplittingParams, FeatureParams, TrainingParams, TrainingPipelineParams
from make_dataset import split_data
from make_features import transform_data, build_transformer
from fit_predict import train_model, predict_model, evaluate_model
from train_pipeline import run_train_pipeline


def create_fake_frame(num_rows: int) -> pd.DataFrame:
    fake = Faker()
    data = fake.csv(data_columns=("{{name}}", "{{address}}"), num_rows=num_rows).split('\r\n')
    data = [row[1:-1].split('\",\"') for row in data][:-1]
    df = pd.DataFrame.from_records(data, )
    return df


class TestReadData(TestCase):
    def test_split_data(self):
        num_rows = 100
        params = SplittingParams(test_size=0.3, random_state=42, shuffle=False)
        test_size = int(num_rows * params.test_size)
        df = create_fake_frame(num_rows)
        df_train_true = df[:-test_size]
        df_test_true = df[-test_size:]

        df_train_my, df_test_my = split_data(df, params)
        self.assertTrue(pd.DataFrame.equals(df_train_my, df_train_true))
        self.assertTrue(pd.DataFrame.equals(df_test_my, df_test_true))


class TestMakeDataset(TestCase):

    def test_OHE(self):
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


class TestModel(TestCase):
    def test_train(self):
        params = TrainingParams(model='LogisticRegression', model_params=None, random_state=None)
        data = pd.read_csv('data/raw/heart_cleveland_upload.csv')
        y_train = data['condition'].values
        X_train = data.drop('condition', axis=1).values
        model = train_model(X_train, y_train, params)
        model_true = LogisticRegression()
        model_true.fit(X_train, y_train)
        self.assertListEqual(model_true.coef_[0].tolist(), model.coef_[0].tolist())

    def test_predict(self):
        params = TrainingParams(model='LogisticRegression', model_params=None, random_state=None)
        y_train = np.array([1, 1, 0])
        X_train = np.ones((3, 3)) * y_train.reshape(-1, 1)
        model = train_model(X_train, y_train, params)
        predict_my = predict_model(model, X_train).tolist()
        predict_true = y_train.tolist()
        self.assertListEqual(predict_true, predict_my)

    def test_evaluate(self):
        target = np.array([1, 0, 1, 0, 1, 0])
        predicts = np.array([1, 0, 1, 0, 1, 0])
        scores_true = {'accuracy': 1, 'f1': 1, 'roc_auc': 1}
        scores_my = evaluate_model(predicts, target)
        self.assertDictEqual(scores_true, scores_my)

        target = np.array([1, 0, 1, 0, 1, 0])
        predicts = np.array([0, 1, 0, 1, 0, 1])
        scores_true = {'accuracy': 0, 'f1': 0, 'roc_auc': 0}
        scores_my = evaluate_model(predicts, target)
        self.assertDictEqual(scores_true, scores_my)

        target = np.array([1, 0, 1, 0, 1, 0])
        predicts = np.array([0, 0, 0, 0, 0, 0])
        scores_true = {'accuracy': 0.5, 'f1': 0.0, 'roc_auc': 0.5}
        scores_my = evaluate_model(predicts, target)
        self.assertDictEqual(scores_true, scores_my)


class TestPipeline(TestCase):
    def test_pipeline(self):
        input_path = 'data/raw/heart_cleveland_upload.csv'
        output_path = ''
        metric_path = ''
        splitting_params = SplittingParams(test_size=0.3, random_state=None, shuffle=True)
        feature_params = FeatureParams(numerical_columns=['age', 'oldpeak', 'trestbps', 'thalach', 'chol'],
                                       categorical_columns=['sex', 'restecg', 'slope', 'fbs', 'cp', 'exang', 'thal', 'ca'],
                                       columns_to_drop='',
                                       target_column='condition',
                                       fill_na_numerical_strategy='mean',
                                       fill_na_categorical_strategy='most_frequent')
        training_params = TrainingParams(model='LogisticRegression',
                                         model_params=None,
                                         random_state=None)

        params = TrainingPipelineParams(input_data_path=input_path,
                                        output_model_path=output_path,
                                        metric_path=metric_path,
                                        splitting_params=splitting_params,
                                        feature_params=feature_params,
                                        training_params=training_params)
        run_train_pipeline(params)


if __name__ == "__main__":
    unittest.main()
