import pathlib
import typing as T

from Levenshtein import distance

from aste.data.common import AspectData
from aste.train.data_providers.datasets import BaseDataset


class BaseDecoder:
    @staticmethod
    def decode(file: pathlib.Path, dataset_class: T.Type[BaseDataset]) -> T.Dict[int, T.List[AspectData]]:
        raise NotImplementedError()
    
    @staticmethod
    def nearest_levenshtein_word(text: str, key: str, max_dist: float = 0.2) -> T.Optional[str]:
        text_words = text.split()

        nearest_word = None
        min_dist = 1.0
        for word in text_words:
            current_distance = distance(word.lower(), key.lower())
            if current_distance < min_dist:
                min_dist = current_distance
                nearest_word = word
        
        if min_dist > max_dist:
            return None
        return nearest_word

