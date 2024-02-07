from collections import defaultdict
import pathlib
import typing as T 

from aste.data.common import AspectData
from aste.train.data_providers.datasets import BaseDataset

from .base_decoder import BaseDecoder


class BasicDecoder(BaseDecoder):
    @staticmethod
    def decode(file: pathlib.Path, dataset_class: T.Type[BaseDataset]) -> T.Dict[int, T.Set[AspectData]]:
        decoded_lines = defaultdict(set)
        with open(file, 'r') as input_file:
            lines = input_file.readlines()

        for line in lines:
            sample_id, text, prediction = line.split('\t')
            sample_id = int(sample_id)

            decoded_lines[sample_id].update(dataset_class.decode(text=text, prediction=prediction))

        return decoded_lines
