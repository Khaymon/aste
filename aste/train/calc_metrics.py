import argparse
import pathlib
import typing as T

from aste.data.readers import BankDataReader
from aste.data.common import AspectData, SampleData
from aste.train.data_providers import ModelDataset


def _parse_args():
    parser = argparse.ArgumentParser(
        "Train model",
        "Train model with Lightning",
        add_help=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--test-path", type=pathlib.Path, required=True, help="Path to the test dataset")
    parser.add_argument("--predictions-path", type=pathlib.Path, required=True, help="Path to the predictions")

    return parser.parse_args()


class Metrics:
    def calculate(self, true: T.List[T.List[AspectData]], predicted: T.List[T.List[SampleData]]):
        assert len(true) == len(predicted)

        self.tp = 0
        self.fp = 0
        self.fn = 0
        for true_aspects, pred_aspects in zip(true, predicted):
            current_tp = sum(pred_aspect in true_aspects for pred_aspect in pred_aspects)
            current_fp = len(pred_aspects) - current_tp
            current_fn = sum(true_aspect not in pred_aspects for true_aspect in true_aspects)
            
            self.tp += current_tp
            self.fp += current_fp
            self.fn += current_fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)
    
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / (self.precision + self.recall)


def main():
    args = _parse_args()

    with open(args.predictions_path, 'r') as input_file:
        lines = input_file.readlines()
    
    test_data = BankDataReader.from_file(args.test_path)
    predicted_aspects = [ModelDataset.decode(text) for text in lines]

    metrics = Metrics()
    metrics.calculate([sample.aspects for sample in test_data], predicted_aspects)

    print(metrics.precision, metrics.recall, metrics.f1)


if __name__ == "__main__":
    main()