import pathlib
import typing as T

import aste.data.common as data_common_lib


class BaseDataReader:
    @staticmethod
    def from_file(path: pathlib.Path, train: bool = True) -> T.List[data_common_lib.SampleData]:
        raise NotImplementedError()
