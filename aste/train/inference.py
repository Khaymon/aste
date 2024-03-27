import argparse
import pathlib
import toml
import typing as T

from aste.data.common import AspectData, SampleData
from aste.train.models.model import ASTEModel
from aste.train.models.tasks import BaseGenerativeModel
from aste.train.model_runners import get_model_runner

import torch


def _parse_args():
    parser = argparse.ArgumentParser(
        "Train model",
        "Train model with Lightning",
        add_help=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--train-recipe", type=pathlib.Path, required=True, help="Path to the train recipe")
    parser.add_argument("--inference-recipe", type=pathlib.Path, required=True, help="Path to the train recipe")

    return parser.parse_args()


class Metrics:
    def calculate(self, true: T.List[T.List[AspectData]], predicted: T.List[T.List[SampleData]]):
        assert len(true) == len(predicted)

        self.tp = 0
        self.fp = 0
        self.fn = 0
        for true_aspects, pred_aspects in zip(true, predicted):
            self.tp += sum(pred_aspect in true_aspects for pred_aspect in pred_aspects)
            self.fp += len(pred_aspects) - self.tp
            self.fn += sum(true_aspect not in pred_aspects for true_aspect in true_aspects)

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

    train_recipe = toml.load(args.train_recipe)
    inference_recipe = toml.load(args.inference_recipe)

    device = torch.device("cuda:" + str(inference_recipe["gpu"]) if torch.cuda.is_available() else "cpu")

    print(f"Use {device} for inference")

    model_class = ASTEModel.get_model(train_recipe["model"]["aste_model_name"])
    model = model_class.load_from_checkpoint(inference_recipe["checkpoint"], recipe=train_recipe).to(device)

    model_runner = get_model_runner(inference_recipe["model_runner_name"], inference_recipe, train_recipe)
    model_runner.run(model, inference_recipe["result_path"])


if __name__ == "__main__":
    main()
