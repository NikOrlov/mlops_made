from typing import Union, Dict
import json
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from entities import TrainingParams


SklearnModel = Union[LogisticRegression]


def train_model(data: np.ndarray, target: np.ndarray, params: TrainingParams) -> SklearnModel:
    if params.model_params is None:
        params.model_params = {}
    if params.model == 'LogisticRegression':
        model = LogisticRegression(**params.model_params)
    elif params.model == 'RandomForestClassifier':
        model = RandomForestClassifier(**params.model_params)
    else:
        raise NotImplementedError

    model.fit(data, target)
    return model


def predict_model(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(data)
    return predicts


def evaluate_model(predicts: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if predicts.shape != target.shape:
        raise RuntimeError
    scores = {'accuracy': round(accuracy_score(target, predicts), 4),
              'f1': round(f1_score(target, predicts), 4),
              'roc_auc': round(roc_auc_score(target, predicts), 4)}
    return scores


def build_inference_pipeline(transforms: ColumnTransformer, model: SklearnModel) -> Pipeline:
    pipe = Pipeline([('transforms', transforms),
                     ('model', model)])
    return pipe


def serialize(obj: object, path: str, data_type: str = 'pickle'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if data_type == 'pickle':
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    elif data_type == 'json':
        with open(path, 'w') as file:
            json.dump(obj, file)
    else:
        raise NotImplementedError
