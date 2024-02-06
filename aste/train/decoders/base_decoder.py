import pathlib
import typing as T

from aste.data.common import AspectData
from aste.train.data_providers.datasets import BaseDataset


class BaseDecoder:
    @staticmethod
    def decode(file: pathlib.Path, dataset_class: T.Type[BaseDataset]) -> T.Dict[int, T.List[AspectData]]:
        raise NotImplementedError()
