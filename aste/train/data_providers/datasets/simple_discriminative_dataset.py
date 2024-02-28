import random
import numpy as np
import typing as T

from transformers import PreTrainedTokenizer

from aste.data.common import SampleData
from aste.train.data_providers.datasets import BaseDiscriminativeDataset


class SimpleDiscriminativeDataset(BaseDiscriminativeDataset):
    ASPECT_COMP_SEP = ";"

    def __init__(
        self,
        data: T.List[SampleData],
        false_generated_data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        **kwargs,
    ):
        concated_data = data + false_generated_data
        self.targets = [1] * len(data) + [0] * len(false_generated_data)
        result_data = []
        for sample in concated_data:
            for aspect in sample.aspects:
                result_data.append(SampleData(sample.sample_id, sample.text, [aspect]))

        shuffled_ids = np.arange(len(concated_data))
        random.shuffle(shuffled_ids)

        self.targets = np.array(self.targets)[shuffled_ids]

        super().__init__(np.array(result_data)[shuffled_ids], tokenizer, max_length)

    def __getitem__(self, index: int):
        aspects = self._data[index].aspects
        target = self.targets[index]
        if len(aspects) == 0:
            aspect = ''
        else:
            aspect = aspects[0]
            aspect = aspect.aspect + self.ASPECT_COMP_SEP + aspect.opinion + self.ASPECT_COMP_SEP + aspect.polarity
        
        text = aspect + self._tokenizer.sep_token + self._data[index].text

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
