from collections import defaultdict
import pathlib
import typing as T 

from aste.data.common import AspectData
from aste.train.data_providers.datasets import BaseDataset

from .base_decoder import BaseDecoder


class BasicDecoder(BaseDecoder):
    @staticmethod
    def decode(file: pathlib.Path, dataset_class: T.Type[BaseDataset]) -> T.Dict[int, T.List[AspectData]]:
        decoded_lines = defaultdict(list)
        with open(file, 'r') as input_file:
            lines = input_file.readlines()

        for line in lines:
            sample_id, prediction = line.split('\t')
            sample_id = int(sample_id)

            decoded_lines[sample_id].extend(dataset_class.decode(prediction))

        return decoded_lines
