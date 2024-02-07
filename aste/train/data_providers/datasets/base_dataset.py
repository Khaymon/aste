import typing as T

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from aste.data.common import AspectData, SampleData


class BaseDataset(Dataset):
    def __init__(
        self,
        data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        source_max_length: int,
        target_max_length: int,
        **kwargs
    ):
        self._data = data
        self._tokenizer = tokenizer
        self._source_max_length = source_max_length
        self._target_max_length = target_max_length
        
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int):
        raise NotImplementedError()
    
    @classmethod
    def decode(cls, *, text: str, prediction: str, **kwargs) -> T.List[AspectData]:
        raise NotImplementedError()