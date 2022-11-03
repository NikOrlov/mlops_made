import logging.config
from make_dataset import read_data, split_data
from entities import TrainingPipelineParams
from make_features import build_transformer, split_features_target
from fit_predict import train_model, predict_model, evaluate_model, inference_pipeline
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

    X_train = column_transformer.fit_transform(df_features_train)
    y_train = df_target_train.values
    logger.debug(f'Train shape: {X_train.shape}')
    logger.info(f'Training params {params.training_params} params')

    model = train_model(X_train, y_train, params.training_params)
    inference_pipe = inference_pipeline(column_transformer, model)

    predicts_on_train = predict_model(inference_pipe, X_train)
    train_score = evaluate_model(predicts_on_train, y_train)
    logger.info(f'Train scores: {train_score}')

    X_test = column_transformer.transform(df_features_test)
    y_test = df_target_test.values
    logger.debug(f'Test shape: {X_test.shape}')

    predicts_on_test = predict_model(model, X_test)
    test_score = evaluate_model(predicts_on_test, y_test)
    logger.info(f'Test scores: {test_score}\n')


if __name__ == '__main__':
    params = 3