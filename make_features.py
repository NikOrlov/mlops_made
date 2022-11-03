import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from entities import FeatureParams
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple


def build_categorical_pipeline(params: FeatureParams) -> Pipeline:
    pipe = Pipeline([('impute', SimpleImputer(strategy=params.fill_na_categorical_strategy)),
                     ('OHE', OneHotEncoder())])
    return pipe


def build_numerical_pipeline(params: FeatureParams) -> Pipeline:
    pipe = Pipeline([('impute', SimpleImputer(strategy=params.fill_na_numerical_strategy)),
                     ('normalize', StandardScaler())])
    return pipe


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    categorical_pipe = build_categorical_pipeline(params)
    numerical_pipe = build_numerical_pipeline(params)

    data_transformer = ColumnTransformer([
                                        ("categorical features", categorical_pipe, params.categorical_columns),
                                        ("numerical features", numerical_pipe, params.numerical_columns)
                                        ])
    return data_transformer


def split_features_target(data: pd.DataFrame, params: FeatureParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = data[params.target_column]
    features = data.drop(params.target_column, axis=1)
    return features, target


def transform_data(data: pd.DataFrame, transformer: ColumnTransformer) -> np.ndarray:
    return transformer.fit_transform(data)
