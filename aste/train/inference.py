import argparse
import itertools
import pathlib
import random
from tqdm import tqdm
import typing as T

from aste.data.common import AspectData, SampleData
from aste.train.data_providers import DataModule
from aste.train.models import BaseGenerationModel
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

    model = BaseGenerationModel.load_from_checkpoint(args.checkpoint, recipe=recipe).eval()
    model.to(device)

    output_max_length = recipe.test_dataloader_recipe.dataset_recipe.output_max_length

    tokenizer = getattr(transformers, recipe.tokenizer_class_name).from_pretrained(recipe.model_name)
    all_predictions = []
    all_texts = []
    all_sample_ids = []
    with torch.no_grad():
        orders = list(itertools.permutations(['A', 'O', 'P']))
        random.shuffle(orders)
        for order in orders[:3]:
            test_dataloader = DataModule.get_dataloader(recipe.test_dataloader_recipe, order=order)
            for batch in tqdm(test_dataloader):
                output_ids = model._model.generate(batch["source_ids"].to(device), max_length=output_max_length).cpu()

                texts = tokenizer.batch_decode(batch["source_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                sample_ids = list(batch["sample_ids"].numpy())

                assert len(predictions) == len(sample_ids), f"{len(predictions)} vs {len(sample_ids)}"

                all_predictions.extend(predictions)
                all_texts.extend(texts)
                all_sample_ids.extend(sample_ids)

    with open(args.result_path, 'w') as output_file:
        output_file.writelines(
            '\n'.join(str(sample_id) + '\t' + text + '\t' + prediction
            for sample_id, text, prediction in zip(all_sample_ids, all_texts, all_predictions))
        )


if __name__ == "__main__":
    main()
