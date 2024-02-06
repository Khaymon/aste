from aste.data.readers import *


class DataReader:
    @staticmethod
    def get_reader(name: str) -> BaseDataReader:
        if name == "BankDataReader":
            return BankDataReader
        elif name == "SentenceBankDataReader":
            return SentenceBankDataReader
        else:
            raise ValueError(f"Unknown data reader {name}")
