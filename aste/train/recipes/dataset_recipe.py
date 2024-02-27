from dataclasses import dataclass
import typing as T

from .base_recipe import BaseRecipe


@dataclass
class DatasetRecipe(BaseRecipe):
    DATASET_KEY: T.ClassVar[str] = "dataset"

    data_path: str
    datareader_class_name: str
    dataset_class_name: str

    tokenizer_class_name: str
    tokenizer_model_name: str

    input_max_length: int = 512
    output_max_length: int = 128

    max_length: int = 512

    @classmethod
    def from_dict(cls, values: T.Dict) -> "DatasetRecipe":
        return cls(**values[cls.DATASET_KEY])
