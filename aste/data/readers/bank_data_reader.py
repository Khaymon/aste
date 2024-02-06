import ast
import pathlib
import typing as T

import aste.data.common as data_common_lib

from .base_data_reader import BaseDataReader


class BankDataReader(BaseDataReader):
    TARGET_SEPARATOR = "####"

    @staticmethod
    def _str_from_ids(text: str, ids: T.List[int]) -> str:
        return ' '.join(text.split()[idx] for idx in ids)

    @staticmethod
    def from_file(path: pathlib.Path, train: bool = True) -> T.List[data_common_lib.SampleData]:
        with open(path, 'r') as input_file:
            lines = input_file.readlines()

        data = []
        for sample_id, line in enumerate(lines):
            text, aspects = line.split(BankDataReader.TARGET_SEPARATOR)
            
            if train:
                aspects = [
                    data_common_lib.AspectData(
                        aspect=BankDataReader._str_from_ids(text, aspect_ids),
                        opinion=BankDataReader._str_from_ids(text, opinion_ids),
                        polarity=polarity,
                        aspect_ids=aspect_ids,
                        opinion_ids=opinion_ids,
                    )
                    for aspect_ids, opinion_ids, polarity in ast.literal_eval(aspects)
                ]
            else:
                aspects = None

            data.append(data_common_lib.SampleData(sample_id=sample_id, text=text, aspects=aspects))

        return data
