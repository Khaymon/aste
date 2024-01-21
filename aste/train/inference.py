import argparse
import pathlib
from tqdm import tqdm
import typing as T

from aste.data.readers import BankDataReader
from aste.data.common import AspectData, SampleData
from aste.train.data_providers import ModelDataset, ModelDatasetModule
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
    parser.add_argument("--data-path", type=pathlib.Path, required=True, help="Path to the dataset")
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

    data = BankDataReader.from_file(args.data_path)
    dataset_module = ModelDatasetModule(recipe=recipe, dev_data=data)

    model = BaseGenerationModel.load_from_checkpoint(args.checkpoint, recipe=recipe).eval()
    model.to(device)

    tokenizer = getattr(transformers, recipe.tokenizer_class_name).from_pretrained(recipe.model_name)
    results = []
    with torch.no_grad():
        for batch in tqdm(dataset_module.val_dataloader()):
            output_ids = model._model.generate(batch["source_ids"].to(device), max_length=recipe.output_max_length).cpu()
            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            results.extend(texts)
    
    with open(args.result_path, 'w') as output_file:
        output_file.writelines('\n'.join(results))


if __name__ == "__main__":
    main()
