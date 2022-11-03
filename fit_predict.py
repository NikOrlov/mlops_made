import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from entities import TrainingParams
from typing import Union, Dict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


SklearnModel = Union[LogisticRegression]


def train_model(data: np.ndarray, target: np.ndarray, params: TrainingParams) -> SklearnModel:
    if params.model == 'LogisticRegression':
        model = LogisticRegression()
    elif params.model == 'RandomForestClassifier':
        model = RandomForestClassifier()
    else:
        raise NotImplementedError

    model.fit(data, target)
    return model


def predict_model(model: Pipeline, data: np.ndarray) -> np.ndarray:
    predicts = model.predict(data)
    return predicts


def evaluate_model(predicts: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if predicts.shape != target.shape:
        raise RuntimeError
    scores = {'accuracy': round(accuracy_score(target, predicts), 4),
              'f1': round(f1_score(target, predicts), 4),
              'roc_auc': round(roc_auc_score(target, predicts), 4)}
    return scores


def inference_pipeline(transforms: ColumnTransformer, model: SklearnModel) -> Pipeline:
    pipe = Pipeline([('transforms', transforms),
                     ('model', model)])
    return pipe
