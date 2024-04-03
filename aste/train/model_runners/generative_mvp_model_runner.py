from copy import deepcopy
import itertools
import random
import pathlib
from tqdm import tqdm
import typing as T

import transformers

from aste.train.models.tasks import BaseGenerativeModel
from aste.train.data_providers import DataModule

from .model_runner import ModelRunner


class GenerativeMVPModelRunner(ModelRunner):
    def __init__(self, train_recipe: T.Dict, inference_recipe: T.Dict):
        super().__init__(train_recipe, inference_recipe)

        self._tokenizer = getattr(
            transformers, train_recipe["model"]["hub_tokenizer_name"]
        ).from_pretrained(train_recipe["model"]["hub_tokenizer_checkpoint"])

    def _run(self, model: BaseGenerativeModel, result_path: pathlib.Path, **kwargs):
        orders = list(itertools.permutations(['A', 'O', 'P']))
        random.shuffle(orders)

        all_predictions = []
        all_texts = []
        all_sample_ids = []
        for order in orders[:self._inference_recipe["n_orders"]]:
            current_inference_recipe = deepcopy(self._inference_recipe)
            current_inference_recipe["dataloader"]["dataset"]["order"] = order
            test_dataloader = DataModule.get_dataloader(self._train_recipe["model"], self._inference_recipe["dataloader"])

            for batch in tqdm(test_dataloader):
                output_ids = model._model.generate(
                    batch["source_ids"].to(model.device),
                    max_length=self._inference_recipe["dataloader"]["dataset"]["output_max_length"],
                    **self._inference_recipe.get("generation", {})
                ).cpu()

                texts = self._tokenizer.batch_decode(batch["source_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                predictions = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                sample_ids = list(batch["sample_ids"].numpy())

                assert len(predictions) == len(sample_ids), f"{len(predictions)} vs {len(sample_ids)}"

                all_predictions.extend(predictions)
                all_texts.extend(texts)
                all_sample_ids.extend(sample_ids)

        with open(result_path, 'w') as output_file:
            output_file.writelines(
                '\n'.join(str(sample_id) + '\t' + text + '\t' + prediction
                for sample_id, text, prediction in zip(all_sample_ids, all_texts, all_predictions))
            )