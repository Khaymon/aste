import typing as T

from transformers import PreTrainedTokenizer

from aste.data.common import SampleData
from aste.train.data_providers.datasets import BaseDataset


class BaseGenerativeDataset(BaseDataset):
    def __init__(
        self,
        data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        source_max_length: int,
        target_max_length: int,
        **kwargs,
    ):
        super().__init__(data, tokenizer)
        self._source_max_length = source_max_length
        self._target_max_length = target_max_length
