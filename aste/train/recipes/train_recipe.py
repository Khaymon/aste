from dataclasses import dataclass
import typing as T

from .base_recipe import BaseRecipe
from .dataloader_recipe import DataLoaderRecipe


@dataclass
class TrainRecipe(BaseRecipe):
    DATALOADERS_KEY: T.ClassVar[str] = "dataloaders"

    TRAIN_KEY: T.ClassVar[str] = "train"
    DEV_KEY: T.ClassVar[str] = "dev"
    TEST_KEY: T.ClassVar[str] = "test"

    train_dataloader_recipe: DataLoaderRecipe
    dev_dataloader_recipe: DataLoaderRecipe
    test_dataloader_recipe: DataLoaderRecipe

    model_class_name: str
    tokenizer_class_name: str
    model_name: str

    freeze: int = 0

    epochs: int = 5

    @classmethod
    def from_dict(cls, values: T.Dict) -> "TrainRecipe":
        train_dataloader_recipe = DataLoaderRecipe.from_dict(values[cls.DATALOADERS_KEY][cls.TRAIN_KEY])
        dev_dataloader_recipe = DataLoaderRecipe.from_dict(values[cls.DATALOADERS_KEY][cls.DEV_KEY])
        test_dataloader_recipe = DataLoaderRecipe.from_dict(values[cls.DATALOADERS_KEY][cls.TEST_KEY])

        return cls(
            train_dataloader_recipe=train_dataloader_recipe,
            dev_dataloader_recipe=dev_dataloader_recipe,
            test_dataloader_recipe=test_dataloader_recipe,
            **values[cls.TRAIN_KEY],
        )
