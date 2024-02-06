import random
import typing as T

from transformers import PreTrainedTokenizer

from aste.data.common import AspectData, SampleData

from .base_dataset import BaseDataset


class ModelDataset(BaseDataset):
    ASPECT_COMP_SEP = ";"
    ASPECTS_SEP = "|"
    ORDER_SEP = "<O>"

    ASPECT_TOKEN = "A"
    OPINION_TOKEN = "O"
    POLARITY_TOKEN = "P"
    ASPECTS_TOKENS = [ASPECT_TOKEN, OPINION_TOKEN, POLARITY_TOKEN]

    def __init__(
        self,
        data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        source_max_length: int,
        target_max_length: int,
        order: T.Optional[T.List[str]] = None,
    ):
        super().__init__(data, tokenizer, source_max_length, target_max_length)

        self.order = order

    def __getitem__(self, index: int):
        if self.order:
            order = self.order
        else:
            order = self.ASPECTS_TOKENS.copy()
            random.shuffle(order)
        assert len(set(order)) == 3, "Length of distinct tokens must be equal to 3"
        
        text = self._tokenizer.bos_token + self._data[index].text + self.ORDER_SEP + ','.join(order) + self._tokenizer.eos_token
        
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
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
        }
    
    @classmethod
    def decode(cls, text: str, prediction: str) -> T.List[AspectData]:
        try:
            text, order = text.split(cls.ORDER_SEP)
            decoded_aspects = []
            for aspect_tuple in prediction.split(cls.ASPECTS_SEP):
                triplet = aspect_tuple.split(cls.ASPECT_COMP_SEP)
                assert len(triplet) == 3

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
