from aste.train.data_providers.datasets import *


class ASTEDataset:
    @staticmethod
    def get_dataset(name: str):
        if name == "BasicDataset":
            return BasicDataset
        else:
            raise ValueError(f"Dataset {name} is not implemented")
