from .base_generation_model import BaseGenerationModel

from transformers import T5ForConditionalGeneration


class T5Model(BaseGenerationModel):
    def __init__(self, model_name: str, tokenizer = None, freeze = 40):
        super().__init__()

        self._model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        self._tokenizer = tokenizer

        for parameter_num, parameter in enumerate(self._model.parameters()):
            if parameter_num < freeze:
                parameter.requires_grad = False
