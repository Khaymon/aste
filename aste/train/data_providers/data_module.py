import transformers
from torch.utils.data import Dataset, DataLoader

from aste.data.reader import DataReader
from aste.train.recipes import DatasetRecipe, DataLoaderRecipe
from aste.train.data_providers.dataset import ASTEDataset


class DataModule:
    @staticmethod
    def get_dataset(dataset_recipe: DatasetRecipe) -> Dataset:
        return ASTEDataset.get_dataset(dataset_recipe.dataset_class_name)(
            data=DataReader.get_reader(dataset_recipe.datareader_class_name).from_file(dataset_recipe.data_path),
            tokenizer=getattr(transformers, dataset_recipe.tokenizer_class_name).from_pretrained(dataset_recipe.tokenizer_model_name),
            source_max_length=dataset_recipe.input_max_length,
            target_max_length=dataset_recipe.output_max_length,
        )

    @staticmethod
    def get_dataloader(dataloader_recipe: DataLoaderRecipe) -> DataLoader:
        dataset = DataModule.get_dataset(dataloader_recipe.dataset_recipe)

        return DataLoader(
            dataset,
            batch_size=dataloader_recipe.batch_size,
            num_workers=dataloader_recipe.num_workers,
        )
