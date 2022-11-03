import logging.config
import pickle

import pandas as pd
import numpy as np
from make_dataset import read_data, split_data
from entities import TrainingPipelineParams
from make_features import build_transformer, transform_data, split_features_target
from fit_predict import train_model, predict_model, evaluate_model, build_inference_pipeline, serialize
from logger import log_conf


logging.config.dictConfig(log_conf)
logger = logging.getLogger('file_stream')


def run_train_pipeline(params: TrainingPipelineParams):
    logger.debug('Start run')
    df = read_data(params.input_data_path)
    logger.debug(f'Dataset shape: {df.shape}')

    df_train, df_test = split_data(df, params.splitting_params)
    df_features_train, df_target_train = split_features_target(df_train, params.feature_params)
    df_features_test, df_target_test = split_features_target(df_test, params.feature_params)
    logger.debug(f'Train set size: {len(df_train)}, test set size: {len(df_test)}')

    column_transformer = build_transformer(params.feature_params)
    X_train = transform_data(df_features_train, column_transformer)
    y_train = df_target_train.values
    logger.debug(f'Train shape: {X_train.shape}')
    logger.info(f'Training params {params.training_params} params')

    model = train_model(X_train, y_train, params.training_params)
    inference_pipe = build_inference_pipeline(column_transformer, model)

    predicts_on_train = predict_model(inference_pipe, df_features_train)
    train_score = evaluate_model(predicts_on_train, y_train)
    logger.info(f'Train scores: {train_score}')

    y_test = df_target_test.values
    logger.debug(f'Test shape: {df_features_test.shape}')

    predicts_on_test = predict_model(inference_pipe, df_features_test)
    test_score = evaluate_model(predicts_on_test, y_test)
    logger.info(f'Test scores: {test_score}\n')


def train_pipeline(params: TrainingPipelineParams):
    logger.debug('Start training!')
    data = read_data(params.input_data_path)
    df_train, df_test = split_data(data, params.splitting_params)
    logger.debug(f'Train set size: {len(df_train)}, test set size: {len(df_test)}')

    df_features_train, df_target_train = split_features_target(df_train, params.feature_params)
    column_transformer = build_transformer(params.feature_params)
    X_train = transform_data(df_features_train, column_transformer)
    y_train = df_target_train.values
    model = train_model(X_train, y_train, params.training_params)
    inference_pipeline = build_inference_pipeline(column_transformer, model)

    predictions_train = predict_model(inference_pipeline, df_features_train)
    score_train = evaluate_model(predictions_train, y_train)
    logger.info(f'Train score: {score_train}')

    df_features_test, df_target_test = split_features_target(df_test, params.feature_params)
    predictions_test = predict_model(inference_pipeline, df_features_test)
    score_test = evaluate_model(predictions_test, df_target_test.values)
    logger.info(f'Test score: {score_test}')

    scores = {'train': score_train, 'test': score_test}
    serialize(scores, params.metric_path, 'json')
    serialize(inference_pipeline, params.output_model_path, 'pickle')


def predict_pipeline(pipeline_path: str, data: pd.DataFrame, output_path: str) -> np.ndarray:
    with open(pipeline_path, 'rb') as file:
        pipe = pickle.load(file)
    predictions = pipe.predict(data)
    with open(output_path, 'w') as file:
        file.write('\n'.join([str(y) for y in predictions]))
    return predictions


if __name__ == '__main__':
    params = 3