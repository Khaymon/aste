import pathlib

from aste.data.readers import BankDataReader
from aste.data.common import AspectData, SampleData


def test_read_from_file_train():
    path = "./test_data/bank_train_sample_data.txt"
    
    got = BankDataReader.from_file(path)
    want = [
        SampleData(
            text="Небольшой пример текста , который может быть написан . Даже из двух предложений . . .",
            aspects=[AspectData([182], [184, 185], "NEG")]
        ),
        SampleData(
            text="И еще один небольшой пример .",
            aspects=[AspectData([2], [1], "NEG"), AspectData([226], [225], "NEG")]
        )
    ]
    
    assert got == want


def test_read_from_file_test():
    path = "./test_data/bank_train_sample_data.txt"
    
    got = BankDataReader.from_file(path, train=False)
    want = [
        SampleData(
            text="Небольшой пример текста , который может быть написан . Даже из двух предложений . . ."
        ),
        SampleData(
            text="И еще один небольшой пример ."
        ),
    ]
    
    assert got == want
