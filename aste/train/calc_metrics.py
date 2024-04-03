import argparse
import pathlib
import typing as T

from aste.data.readers import BankDataReader
from aste.data.common import AspectData, SampleData
from aste.train.decoders import BasicDecoder, MVPDecoder
from aste.train.data_providers.dataset import ASTEDataset


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
    def _drop_duplicates(self, samples: T.Dict[int, T.Set[AspectData]]) -> T.Dict[int, T.Set[AspectData]]:
        return {sample_id: set(aspects) for sample_id, aspects in samples.items()}

    def calculate(self, true: T.Dict[int, T.Set[AspectData]], predicted: T.Dict[int, T.Set[AspectData]]):
        true = self._drop_duplicates(true)
        predicted = self._drop_duplicates(predicted)

        assert len(true) == len(predicted)

        self.tp = 0
        self.fp = 0
        self.fn = 0
        for sample_id in (set(true) | set(predicted)):
            if sample_id not in true:
                raise ValueError(f"Sample id {sample_id} is not found in true aspects")
            if sample_id not in predicted:
                raise ValueError(f"Sample id {sample_id} is not found in predicted aspects")

            true_aspects = true[sample_id]
            pred_aspects = predicted[sample_id]

            current_tp = sum(pred_aspect in true_aspects for pred_aspect in pred_aspects)
            current_fp = len(pred_aspects) - current_tp
            current_fn = sum(true_aspect not in pred_aspects for true_aspect in true_aspects)

            print("################")
            print(f"True aspects: {true_aspects}")
            print("-------------------")
            print(f"Predicted aspects: {pred_aspects}")
            print("-------------------")
            pred_diff = set(pred_aspect for pred_aspect in pred_aspects if pred_aspect not in true_aspects)
            print(f"Predicted aspects not in true aspects {pred_diff}")
            print("-------------------")
            true_diff = set(true_aspect for true_aspect in true_aspects if true_aspect not in pred_aspects)
            print(f"True aspects not in pred aspects {true_diff}")
            
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

    test_data = BankDataReader.from_file(args.test_path)
    test_aspects = {sample.sample_id: sample.aspects for sample in test_data}

    predicted_aspects = MVPDecoder.decode(args.predictions_path, votes=3)

    metrics = Metrics()
    metrics.calculate(test_aspects, predicted_aspects)

    print(f"Precision: {metrics.precision}, Recall: {metrics.recall}, F1: {metrics.f1}")


if __name__ == "__main__":
    main()