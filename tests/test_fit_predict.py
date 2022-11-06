from unittest import TestCase
import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.entities import TrainingParams
from src.fit_predict import train_model, predict_model, evaluate_model
from tests.data_generator import fake_dataset_builder


class TestModel(TestCase):
    def test_train(self):
        params = TrainingParams(model='LogisticRegression', model_params=None, random_state=None)
        data = fake_dataset_builder.generate_dataset()
        y_train = data['condition'].values
        x_train = data.drop('condition', axis=1).values
        model = train_model(x_train, y_train, params)
        model_true = LogisticRegression()
        model_true.fit(x_train, y_train)
        self.assertListEqual(model_true.coef_[0].tolist(), model.coef_[0].tolist())

    def test_predict(self):
        params = TrainingParams(model='LogisticRegression', model_params=None, random_state=None)
        y_train = np.array([1, 1, 0])
        x_train = np.ones((3, 3)) * y_train.reshape(-1, 1)
        model = train_model(x_train, y_train, params)
        predict_my = predict_model(model, x_train).tolist()
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


if __name__ == '__main__':
    unittest.main()
