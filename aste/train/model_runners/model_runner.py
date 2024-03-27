import pathlib
import typing as T

import torch

from aste.train.models.tasks import BaseModel


class ModelRunner:
    def __init__(self, train_recipe: T.Dict, inference_recipe: T.Dict):
        self._train_recipe = train_recipe
        self._inference_recipe = inference_recipe

    def _run(self, model: BaseModel, result_path: pathlib.Path, **kwargs):
        raise NotImplementedError()
    
    def run(self, model: BaseModel, result_path: pathlib.Path, **kwargs):
        with torch.no_grad():
            self._run(model, result_path, **kwargs)
