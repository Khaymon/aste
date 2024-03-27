import typing as T

import transformers
from torch.utils.data import Dataset, DataLoader

from aste.data.reader import DataReader
from aste.train.data_providers.dataset import ASTEDataset


class DataModule:
    @staticmethod
    def get_dataset(model_recipe: T.Dict[str, T.Any], dataset_recipe: T.Dict[str, T.Any]) -> Dataset:
        return ASTEDataset.get_dataset(dataset_recipe["dataset_class_name"])(
            data=DataReader.get_reader(dataset_recipe["datareader_class_name"]).from_file(dataset_recipe["data_path"]),
            tokenizer=getattr(transformers, model_recipe["hub_tokenizer_name"]).from_pretrained(model_recipe["hub_tokenizer_checkpoint"]),
            source_max_length=dataset_recipe["input_max_length"],
            target_max_length=dataset_recipe["output_max_length"],
            **dataset_recipe,
        )

    @staticmethod
    def get_dataloader(model_recipe: T.Dict[str, T.Any], dataloader_recipe: T.Dict[str, T.Any]) -> DataLoader:
        return DataLoader(
            DataModule.get_dataset(model_recipe, dataloader_recipe["dataset"]),
            batch_size=dataloader_recipe["dataloader"]["batch_size"],
            num_workers=dataloader_recipe["dataloader"]["num_workers"],
        )
