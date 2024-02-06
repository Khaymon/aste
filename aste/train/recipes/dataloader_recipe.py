from dataclasses import dataclass
import typing as T

from .base_recipe import BaseRecipe
from .dataset_recipe import DatasetRecipe


@dataclass
class DataLoaderRecipe(BaseRecipe):
    DATALOADER_KEY: T.ClassVar[str] = "dataloader"

    dataset_recipe: DatasetRecipe

    batch_size: int
    num_workers: int

    @classmethod
    def from_dict(cls, values: T.Dict) -> "DataLoaderRecipe":
        dataset_recipe = DatasetRecipe.from_dict(values)

        return cls(dataset_recipe=dataset_recipe, **values[cls.DATALOADER_KEY])
