from aste.data.readers import BankDataReader, SentenceBankDataReader
from aste.data.common import AspectData, SampleData


def test_read_from_file_train():
    path = "./test_data/bank_train_sample_data.txt"
    
    got = BankDataReader.from_file(path)
    want = [
        SampleData(
            text="Небольшой пример текста , который может быть написан . Даже из двух предложений . . .",
            aspects=[AspectData(aspect="пример текста", opinion="Небольшой", polarity="NEG", aspect_ids=[1, 2], opinion_ids=[0])]
        ),
        SampleData(
            text="И еще один небольшой пример .",
            aspects=[
                AspectData(aspect="пример", opinion="небольшой", polarity="NEG", aspect_ids=[4], opinion_ids=[3]),
                AspectData(aspect="И", opinion="еще", polarity="POS", aspect_ids=[0], opinion_ids=[1]),
            ]
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


def test_sentences_read_from_file():
    path = "./test_data/real_bank_train_sample.txt"

    got = SentenceBankDataReader.from_file(path)
    want = [
        SampleData(
            text="Номер 448035 , отличный сотрудник , сделал все быстро и в срок , как и обещал , большое спасибо !",
            aspects=[AspectData(aspect="сотрудник", opinion="отличный", polarity="POS", aspect_ids=[4], opinion_ids=[3])],
        ),
        SampleData(
            text="Суть проблемы была в том , что мне было необходимо получить договора на мои карты , которые я потеряо или забыл давным давно , сотрудник понял суть вопроса с первой минуты , сделал запрос коллегам и они выслали все в течение 30 минут !",
            aspects=[],
        ),
        SampleData(
            text="В ноябре обращался в банк ВТБ , в офис на Мира , 26 .",
            aspects=[],
        ),
        SampleData(
            text="Мне необходимо было установить приложение на телефон .",
            aspects=[],
        ),
        SampleData(
            text="Старое приложение не работало , а новое скачать не получалось .",
            aspects=[],
        ),
        SampleData(
            text="Сотрудники подсказали как и где его скачать , помогли установить .",
            aspects=[],
        ),
        SampleData(
            text="Очень отзывчивые , спасибо .",
            aspects=[],
        ),
        SampleData(
            text="Рекомендую , персонал очень отзывчивый , доброжелательный .",
            aspects=[
                AspectData(aspect="персонал", opinion="очень отзывчивый", polarity="POS", aspect_ids=[51], opinion_ids=[52, 53]),
                AspectData(aspect="персонал", opinion="доброжелательный", polarity="POS", aspect_ids=[51], opinion_ids=[55]),
            ],
        ),
    ]

    assert got == want
