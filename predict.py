import pickle
import click
from make_dataset import read_data


def predict_pipeline(pipeline_path: str, data_path: str, output_path: str):
    with open(pipeline_path, 'rb') as file:
        train_obj = pickle.load(file)
    pipe = train_obj['pipeline']
    train_config = train_obj['train_config']

    data = read_data(data_path)
    categorical_cols = train_config.feature_params.categorical_columns
    numerical_cols = train_config.feature_params.numerical_columns

    columns = categorical_cols + numerical_cols
    for column in columns:
        if column not in data.columns:
            raise KeyError(f'Feature {column} does not exist in input data')
    data = data[columns]
    predictions = pipe.predict(data)
    with open(output_path, 'w') as file:
        file.write('\n'.join([str(y) for y in predictions]))


@click.command()
@click.argument('model_path')
@click.argument('data_path')
@click.argument('output_path')
def predict_pipeline_command(model_path: str, data_path: str, output_path: str):
    predict_pipeline(model_path, data_path, output_path)


if __name__ == '__main__':
    predict_pipeline_command()
