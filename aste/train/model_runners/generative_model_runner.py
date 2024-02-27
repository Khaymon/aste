from pathlib import Path
from aste.train.models.tasks import BaseGenerativeModel
from .model_runner import ModelRunner


class GenerativeModelRunner(ModelRunner):
    def _run(self, model: BaseGenerativeModel, result_path: Path):
        