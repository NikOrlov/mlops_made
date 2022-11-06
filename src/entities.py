from dataclasses import dataclass
from typing import List, Dict, Union, Optional
ModelParams = Union[str, float, int, None]


@dataclass()
class SplittingParams:
    test_size: float
    random_state: Optional[int]
    shuffle: bool


@dataclass()
class FeatureParams:
    numerical_columns: List[str]
    categorical_columns: List[str]
    columns_to_drop: Optional[List[str]]
    target_column: str
    fill_na_numerical_strategy: str
    fill_na_categorical_strategy: str


@dataclass()
class TrainingParams:
    model: str
    model_params: Optional[Dict[str, ModelParams]]
    random_state: Optional[int]


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    training_params: TrainingParams
