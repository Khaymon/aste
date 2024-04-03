import typing as T

import transformers

from .generative_mvp_model_runner import GenerativeMVPModelRunner
from .model_runner import ModelRunner  # noqa



def get_model_runner(model_runner_name: str, train_recipe: T.Dict, inference_recipe: T.Dict) -> ModelRunner:
    if model_runner_name == "GenerativeMVPModelRunner":
        return GenerativeMVPModelRunner(train_recipe=train_recipe, inference_recipe=inference_recipe)
    else:
        raise NotImplementedError(f"Model runner {model_runner_name} is not implemented")
