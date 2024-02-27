import random
import numpy as np
import typing as T

from transformers import PreTrainedTokenizer

from aste.data.common import SampleData
from aste.train.data_providers.datasets import BaseDiscriminativeDataset


ASPECTS_LEN_DIST = {
    1: 0.9675015852885225,
    2: 0.029486366518706404,
    3: 0.0028535193405199747,
    4: 0.0001585288522511097
}
OPINIONS_LEN_DIST = {
    1: 0.8199112238427394,
    2: 0.14933417882054534,
    3: 0.022986683576410906,
    4: 0.0050729232720355105,
    5: 0.0023779327837666455,
    7: 0.0003170577045022194,
}
POLARITIES_DIST = {
    "NEG": 0.28836398224476856,
    "POS": 0.709575142675967,
    "NEU": 0.002060875079264426,
}


class SimpleDiscriminativeDataset(BaseDiscriminativeDataset):
    ASPECT_COMP_SEP = ";"

    def __init__(
        self,
        data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        negative_proba: float = 0.3,
        aspects_len_dist: T.Optional[T.Dict[int, float]] = ASPECTS_LEN_DIST,
        opinions_len_dist: T.Optional[T.Dict[int, float]] = OPINIONS_LEN_DIST,
        polarities_dist: T.Optional[T.Dict[str, float]] = POLARITIES_DIST,
        **kwargs,
    ):
        super().__init__(data, tokenizer, max_length)

        if negative_proba < 0 or negative_proba > 1:
            raise ValueError(f"Negative proba should be betwee 0 and 1")
        if negative_proba > 0 and any([item is None for item in (aspects_len_dist, opinions_len_dist, polarities_dist)]):
            raise ValueError(f"Should provide distributions for greater than zero negative proba")

        self._negative_proba = negative_proba

        self._aspect_len_dist = aspects_len_dist
        self._opinions_len_dist = opinions_len_dist
        self._polarities_dist = polarities_dist

    def __getitem__(self, index: int):
        aspects = self._data[index].aspects

        target = 0
        chosen_aspect = None
        if aspects:
            if random.uniform(0, 1) > self._negative_proba:
                target = 1
                chosen_aspect = random.choice(aspects)
                chosen_aspect = chosen_aspect.aspect + self.ASPECT_COMP_SEP + \
                                chosen_aspect.opinion + self.ASPECT_COMP_SEP + chosen_aspect.polarity
        if chosen_aspect is None:
            aspect_size = np.random.choice(list(self._aspect_len_dist.keys()), p=list(self._aspect_len_dist.values()))
            opinion_size = np.random.choice(list(self._opinions_len_dist.keys()), p=list(self._opinions_len_dist.values()))
            polarity = np.random.choice(list(self._polarities_dist.keys()), p=list(self._polarities_dist.values()))

            aspect = ' '.join(np.random.choice(self._data[index].text.split(' '), aspect_size, replace=True))
            opinion = ' '.join(np.random.choice(self._data[index].text.split(' '), opinion_size, replace=True))

            chosen_aspect = aspect + self.ASPECT_COMP_SEP + opinion + self.ASPECT_COMP_SEP + polarity
        text = chosen_aspect + self._tokenizer.sep_token + self._data[index].text
        
        tokenized_inputs = self._tokenizer(
            text,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt", 
        )

        ids = tokenized_inputs["input_ids"].flatten()
        mask = tokenized_inputs["attention_mask"].flatten()
        
        return {
            "sample_ids": self._data[index].sample_id,
            "labels": target,
            "input_ids": ids,
            "attention_mask": mask,
        }
