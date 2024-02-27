import pathlib

import torch

from aste.train.models.tasks import BaseModel


class ModelRunner:
    def _run(self, model: BaseModel, result_path: pathlib.Path, **kwargs):
        raise NotImplementedError()
    
    def run(self, model: BaseModel, result_path: pathlib.Path, **kwargs):
        with torch.no_grad():
            self._run(model, result_path, **kwargs)
 