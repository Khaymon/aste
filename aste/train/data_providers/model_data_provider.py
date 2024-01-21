import typing as T

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import transformers
from transformers import PreTrainedTokenizer

from aste.data.common import AspectData, SampleData
from aste.train.recipes import TrainRecipe

from .base_data_provider import BaseDataset


class ModelDataset(BaseDataset):
    ASPECT_COMP_SEP = ";"
    ASPECTS_SEP = "|"

    def __init__(
        self,
        data: T.List[SampleData],
        tokenizer: PreTrainedTokenizer,
        source_max_length: int,
        target_max_length: int
    ):
        super().__init__(data)
        
        self._tokenizer = tokenizer
        self._source_max_length = source_max_length
        self._target_max_length = target_max_length

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


class ModelDatasetModule(pl.LightningDataModule):
    def __init__(self, recipe: TrainRecipe, train_data: T.Optional[T.List[SampleData]] = None, dev_data: T.Optional[T.List[SampleData]] = None):
        self._recipe = recipe

        if train_data:
            self._train_dataset = ModelDataset(
                train_data,
                tokenizer=getattr(transformers, recipe.tokenizer_class_name).from_pretrained(recipe.model_name),
                source_max_length=recipe.input_max_length,
                target_max_length=recipe.output_max_length,
            )
        else:
            self._train_dataset = None
        if dev_data:
            self._dev_dataset = ModelDataset(
                dev_data,
                tokenizer=getattr(transformers, recipe.tokenizer_class_name).from_pretrained(recipe.model_name),
                source_max_length=recipe.input_max_length,
                target_max_length=recipe.output_max_length,
            )
        else:
            self._dev_dataset = None

    def train_dataloader(self) -> DataLoader:
        if not self._train_dataset:
            raise RuntimeError("Train data is not provided")
        return DataLoader(
            self._train_dataset,
            batch_size=self._recipe.train_batch_size,
            shuffle=True,
            num_workers=self._recipe.num_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        if not self._dev_dataset:
            raise RuntimeError("Dev data is not provided")
        return DataLoader(
            self._dev_dataset,
            batch_size=self._recipe.dev_batch_size,
            shuffle=False,
            num_workers=self._recipe.num_workers,
        )
