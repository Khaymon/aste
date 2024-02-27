import typing as T

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from aste.data.common import AspectData, SampleData


class BaseDataset(Dataset):
    def __init__(
        self,
        data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        self._data = data
        self._tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int):
        raise NotImplementedError()
    
    @classmethod
    def decode(cls, *, text: str, prediction: str, **kwargs) -> T.List[AspectData]:
        raise NotImplementedError()