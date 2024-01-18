import typing as T

from transformers import PreTrainedTokenizer

from aste.data.common import SampleData

from .base_data_provider import BaseDataset


class TokenizedDataset(BaseDataset):
    def __init__(self, data: T.List[SampleData], tokenizer: PreTrainedTokenizer, text_max_length: int = 512, target_max_length: int = 128):
        super().__init__(data)
        
        self._tokenizer = tokenizer
        self._text_max_length = text_max_length
        self._target_max_length = target_max_length
        
        self._inputs = []
        self._targets = []
        
        self._build()

    def __getitem__(self, index: int):
        text = self._data[index].text
        aspects = self._data[index].aspects
        
        source_ids = self._inputs[index]["input_ids"].squeeze()
        target_ids = self._targets[index]["input_ids"].squeeze()
        
        source_mask = self._inputs[index]["attenton_mask"].squeeze()
        target_mask = self._targets[index]["attention_mask"].squeeze()
        
        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }
        
    def _build(self):
        for sample in self._data:
            text = sample.text
            aspects = sample.aspects
            
            result_aspects = []
            for aspect in aspects:
                current_aspect = ' '.join(text[index] for index in aspect.aspect)
                current_opinion = ' '.join(text[index] for index in aspect.opinion)
                result_aspects.append(current_aspect + "; " + current_opinion + "; " + aspect.polarity)
            target = self._tokenizer.sep_token.join(result_aspects)
            
            tokenized_inputs = self._tokenizer.batch_encode_plus(
                [text], max_length=self._text_max_length, padding="max_length", return_tensors="pt" 
            )
            
            tokenized_targets = self._tokenizer.batch_encode_plus(
                [target], max_length=self._target_max_length, padding="max_length", return_tensors="pt"
            )
            
            self._inputs.append(tokenized_inputs)
            self._targets.append(tokenized_targets)
