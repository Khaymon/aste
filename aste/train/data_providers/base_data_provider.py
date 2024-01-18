import typing as T

from torch.utils.data import Dataset

from aste.data.common import SampleData


class BaseDataset(Dataset):
    def __init__(self, data: T.List[SampleData], **kwargs):
        self._data = data
        
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int):
        raise NotImplementedError()