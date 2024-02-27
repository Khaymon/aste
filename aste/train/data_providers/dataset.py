from aste.train.data_providers.datasets import *


class ASTEDataset:
    @staticmethod
    def get_dataset(name: str):
        if name == "BasicDataset":
            return SimpleGenerativeDataset
        elif name == "MVPDataset":
            return MVPGenerativeDataset
        elif name == "SimpleDiscriminativeDataset":
            return SimpleDiscriminativeDataset
        else:
            raise ValueError(f"Dataset {name} is not implemented")
