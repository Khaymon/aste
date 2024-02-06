import typing as T

from transformers import PreTrainedTokenizer

from aste.data.common import AspectData, SampleData

from .base_dataset import BaseDataset


class BasicDataset(BaseDataset):
    ASPECT_COMP_SEP = ";"
    ASPECTS_SEP = "|"

    def __init__(
        self,
        data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        source_max_length: int,
        target_max_length: int
    ):
        super().__init__(data, tokenizer, source_max_length, target_max_length)

    def __getitem__(self, index: int):
        text = self._tokenizer.bos_token + self._data[index].text + self._tokenizer.eos_token
        aspects = self._data[index].aspects
        
        result_aspects = []
        for aspect in aspects:
            result_aspects.append(self.ASPECT_COMP_SEP.join((aspect.aspect, aspect.opinion, aspect.polarity)))
        target = self._tokenizer.bos_token + self.ASPECTS_SEP.join(result_aspects) + self._tokenizer.eos_token
        
        tokenized_inputs = self._tokenizer(
            text,
            max_length=self._source_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt", 
        )
        
        tokenized_targets = self._tokenizer(
            target,
            max_length=self._target_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        source_ids = tokenized_inputs["input_ids"].flatten()
        target_ids = tokenized_targets["input_ids"].flatten()
        target_ids[target_ids == self._tokenizer.pad_token_id] = -100
        
        source_mask = tokenized_inputs["attention_mask"].flatten()
        
        return {
            "sample_ids": self._data[index].sample_id,
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
        }
    
    @classmethod
    def decode(cls, text: str) -> T.List[AspectData]:
        decoded_aspects = []
        for aspect_tuple in text.split(cls.ASPECTS_SEP):
            try:
                aspect, opinion, polarity = aspect_tuple.split(cls.ASPECT_COMP_SEP)
                decoded_aspects.append((aspect.strip(), opinion.strip(), polarity.strip()))
            except:
                pass

        return [AspectData(aspect, opinion, polarity) for aspect, opinion, polarity in set(decoded_aspects)]
