import pandas as pd
from entities import SplittingParams
from typing import Tuple
from sklearn.model_selection import train_test_split


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_data(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_set, test_set = train_test_split(data,
                                           test_size=params.test_size,
                                           random_state=params.random_state,
                                           shuffle=params.shuffle)
    return train_set, test_set


