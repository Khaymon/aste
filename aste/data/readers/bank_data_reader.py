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
        for line in lines:
            text, aspects = line.split(BankDataReader.TARGET_SEPARATOR)
            
            if train:
                aspects = [
                    data_common_lib.AspectData(
                        BankDataReader._str_from_ids(text, aspect),
                        BankDataReader._str_from_ids(text, opinion),
                        polarity,
                    )
                    for aspect, opinion, polarity in ast.literal_eval(aspects)
                ]
            else:
                aspects = None

            data.append(data_common_lib.SampleData(text, aspects))

        return data
