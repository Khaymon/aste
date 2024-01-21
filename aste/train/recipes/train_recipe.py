from dataclasses import dataclass
import pathlib
import typing as T

import toml


@dataclass
class TrainRecipe:
    train_path: str
    dev_path: str
    test_path: str

    model_name: str

    model_class_name: T.Optional[str] = None
    tokenizer_class_name: T.Optional[str] = None

    input_max_length: int = 512
    output_max_length: int = 128

    train_batch_size: int = 2
    dev_batch_size: int = 2

    num_workers: int = 4

    freeze: int = 0

    epochs: int = 5

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "TrainRecipe":
        return cls(**toml.load(path))
    