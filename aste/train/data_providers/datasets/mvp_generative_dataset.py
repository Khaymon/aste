from collections import defaultdict
import pathlib
import random
import typing as T

import spacy
from transformers import PreTrainedTokenizer

from aste.data.common import AspectData, SampleData

from .base_dataset import BaseDataset


class MVPGenerativeDataset(BaseDataset):
    ASPECT_COMP_SEP = ";"
    ASPECTS_SEP = "|"
    ORDER_SEP = "<O>"

    ASPECT_TOKEN = "A"
    OPINION_TOKEN = "O"
    POLARITY_TOKEN = "P"
    ASPECTS_TOKENS = [ASPECT_TOKEN, OPINION_TOKEN, POLARITY_TOKEN]

    POS_BLACKLIST = {
        "ADP", "PUNCT", "INTJ", "CCONJ"
    }

    def __init__(
        self,
        data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        source_max_length: int,
        target_max_length: int,
        order: T.Optional[T.List[str]] = None,
        nlp: str = None,
        **kwargs,
    ):
        super().__init__(data, tokenizer)

        self._source_max_length = source_max_length
        self._target_max_length = target_max_length
        if nlp:
            self._nlp = spacy.load(nlp)
        self.order = order

    def __getitem__(self, index: int):
        if self.order:
            order = self.order
        else:
            order = self.ASPECTS_TOKENS.copy()
            random.shuffle(order)
        assert len(set(order)) == 3, "Length of distinct tokens must be equal to 3"

        text = self._data[index].text
        final_tokens = []
        if self._nlp is not None:
            doc = self._nlp(text)
            for token in doc:
                final_tokens.append(str(token))
                pos = str(token.pos_)
                if pos not in self.POS_BLACKLIST:
                    final_tokens.append(pos)
            
            text = ' '.join(final_tokens)
        
        text = self._tokenizer.bos_token + text + self.ORDER_SEP + ','.join(order) + self._tokenizer.eos_token
        
        aspects = self._data[index].aspects
        result_aspects = []
        for aspect in aspects:
            current_aspect = []
            for token in order:
                if token == self.ASPECT_TOKEN:
                    current_aspect.append(aspect.aspect)
                elif token == self.OPINION_TOKEN:
                    current_aspect.append(aspect.opinion)
                elif token == self.POLARITY_TOKEN:
                    current_aspect.append(aspect.polarity)
                else:
                    raise ValueError(f"Unknown token {token}")

            result_aspects.append(self.ASPECT_COMP_SEP.join(current_aspect))
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
    def decode(cls, *, text: str, prediction: str, **kwargs) -> T.List[AspectData]:
        decoded_aspects = []
        try:
            text, order = text.split(cls.ORDER_SEP)
            for aspect_tuple in prediction.split(cls.ASPECTS_SEP):
                triplet = aspect_tuple.split(cls.ASPECT_COMP_SEP)
                if len(triplet) != 3:
                    continue
                aspect = None
                opinion = None
                polarity = None
                for token, order_token in zip(triplet, order.split(',')):
                    if order_token == cls.ASPECT_TOKEN:
                        aspect = token
                    elif order_token == cls.OPINION_TOKEN:
                        opinion = token
                    elif order_token == cls.POLARITY_TOKEN:
                        polarity = token
                    else:
                        raise ValueError(f"Invalid order token {order_token}")

                decoded_aspects.append((aspect.strip(), opinion.strip(), polarity.strip()))
        except:
            pass

        return [AspectData(aspect, opinion, polarity) for aspect, opinion, polarity in set(decoded_aspects)]
    
    @classmethod
    def from_file(cls, *, file_path: pathlib.Path) -> T.Dict[int, T.Set[SampleData]]:
        with open(file_path, 'r') as input_file:
            lines = input_file.readlines()
        
        decoded_file = defaultdict(set)
        for line in lines:
            idx, text, prediction = line.split('\t')
            idx = int(idx)
            aspects = cls.decode(text=text, prediction=prediction)
            text, _ = text.split(cls.ORDER_SEP)

            decoded_file[idx].add(SampleData(sample_id=idx, text=text, aspects=aspects))
        
        return decoded_file
