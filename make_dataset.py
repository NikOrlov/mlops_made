import yaml
import pandas as pd
from entities import SplittingParams, FeatureParams, TrainingParams, TrainingPipelineParams
from typing import Tuple
from sklearn.model_selection import train_test_split


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def read_config(path: str) -> TrainingPipelineParams:
    with open(path) as stream:
        config = yaml.safe_load(stream)
    splitting_params = SplittingParams(**config['splitting_params'])
    feature_params = FeatureParams(**config['feature_params'])
    training_params = TrainingParams(**config['training_params'])
    params = TrainingPipelineParams(input_data_path=config['input_data_path'],
                                    output_model_path=config['output_model_path'],
                                    metric_path=config['metric_path'],
                                    splitting_params=splitting_params,
                                    feature_params=feature_params,
                                    training_params=training_params)
    return params


def split_data(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_set, test_set = train_test_split(data,
                                           test_size=params.test_size,
                                           random_state=params.random_state,
                                           shuffle=params.shuffle)
    return train_set, test_set


