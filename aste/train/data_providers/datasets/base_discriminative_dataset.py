import typing as T

from transformers import PreTrainedTokenizer

from aste.data.common import SampleData

from .base_dataset import BaseDataset


class BaseDiscriminativeDataset(BaseDataset):
    ASPECT_COMP_SEP = ";"

    def __init__(
        self,
        data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        **kwargs,
    ):
        super().__init__(data, tokenizer)

        self._max_length = max_length
