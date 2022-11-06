import logging.config
import click
from make_dataset import read_data, split_data, read_config
from entities import TrainingPipelineParams
from make_features import build_transformer, transform_data, split_features_target, drop_columns
from fit_predict import train_model, \
    predict_model, \
    evaluate_model, \
    build_inference_pipeline, \
    serialize
from logger import log_conf


logging.config.dictConfig(log_conf)
logger = logging.getLogger('file_stream')


def train_pipeline(params: TrainingPipelineParams):
    logger.debug('Start training!')
    logger.debug(f'Params: {params}')
    data = read_data(params.input_data_path)
    drop_columns(data, params.feature_params)
    df_train, df_test = split_data(data, params.splitting_params)
    logger.debug(f'Train set size: {len(df_train)}, test set size: {len(df_test)}')

    df_features_train, df_target_train = split_features_target(df_train, params.feature_params)
    column_transformer = build_transformer(params.feature_params)
    x_train = transform_data(df_features_train, column_transformer)
    y_train = df_target_train.values
    model = train_model(x_train, y_train, params.training_params)
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
    serialize({'pipeline': inference_pipeline, 'train_config': params}, params.output_model_path, 'pickle')


@click.command()
@click.argument('path')
def train_pipeline_command(path: str):
    params = read_config(path)
    train_pipeline(params)


if __name__ == '__main__':
    train_pipeline_command()
