import argparse
from collections import defaultdict
import pathlib
import pickle
import typing as T

from aste.data.readers import BankDataReader
from aste.data.common import AspectData, SampleData
from aste.train.decoders import BasicDecoder
from aste.train.data_providers.dataset import ASTEDataset, MVPGenerativeDataset


def _parse_args():
    parser = argparse.ArgumentParser(
        "Save wrong answers",
        "Save wrong answers from trained model",
        add_help=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--true-path", type=pathlib.Path, required=True, help="Path to the test dataset")
    parser.add_argument("--predicted-path", type=pathlib.Path, required=True, help="Path to the predictions")
    parser.add_argument("--result-path", type=pathlib.Path, required=True, help="Path to store results")

    return parser.parse_args()


def save_wrong_answers(
    true_aspects: T.Dict[int, SampleData],
    predicted_aspects: T.Dict[int, T.Set[SampleData]],
    result_path: pathlib.Path
):
    false_samples = defaultdict(set)
    for idx, sample_bunch in predicted_aspects.items():
        for sample in sample_bunch:
            sample_false_aspects = set()
            for aspect in sample.aspects:
                if aspect not in true_aspects[idx].aspects:
                    sample_false_aspects.add(aspect)
            if len(sample_false_aspects) > 0:
                false_samples[idx].add(SampleData(sample_id=idx, text=sample.text, aspects=list(sample_false_aspects)))
    
    with open(result_path, 'wb') as output_file:
        pickle.dump(false_samples, output_file)


def main():
    args = _parse_args()

    test_data = BankDataReader.from_file(args.true_path)
    true_aspects = {sample.sample_id: sample for sample in test_data}

    predicted_aspects = MVPGenerativeDataset.from_file(file_path=args.predicted_path)

    save_wrong_answers(true_aspects, predicted_aspects, args.result_path)


if __name__ == "__main__":
    main()