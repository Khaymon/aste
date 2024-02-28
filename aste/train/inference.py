import argparse
import pathlib
import typing as T

from aste.data.common import AspectData, SampleData
from aste.train.model_runners import GenerativeMVPModelRunner
from aste.train.models.tasks import BaseGenerativeModel
from aste.train.recipes import TrainRecipe

import torch
import transformers


def _parse_args():
    parser = argparse.ArgumentParser(
        "Train model",
        "Train model with Lightning",
        add_help=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--recipe", type=pathlib.Path, required=True, help="Path to the train recipe")
    parser.add_argument("--gpu", type=int, required=True, help="GPU to inference model")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to the model checkpoint")
    parser.add_argument("--result-path", type=pathlib.Path, required=True, help="Path to the inference result")

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

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    print(f"Use {device} for inference")

    recipe = TrainRecipe.from_file(args.recipe)

    model = BaseGenerativeModel.load_from_checkpoint(args.checkpoint, recipe=recipe).eval()
    model.to(device)

    tokenizer = getattr(transformers, recipe.tokenizer_class_name).from_pretrained(recipe.model_name)

    model_runner = GenerativeMVPModelRunner(tokenizer, recipe.test_dataloader_recipe, n_orders=3)
    model_runner.run(model, args.result_path)


if __name__ == "__main__":
    main()
