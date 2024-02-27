import torch

from aste.train.recipes import TrainRecipe

from .base_model import BaseModel


class BaseGenerativeModel(BaseModel):
    def __init__(self, recipe: TrainRecipe):
        super().__init__(recipe)
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        output = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return output.loss, output.logits
        
    def training_step(self, batch, batch_idx):
        input_ids = batch["source_ids"]
        attention_mask = batch["source_mask"]
        labels = batch["target_ids"]

        loss, _ = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        if (batch_idx + 1) % 50 == 0:
            with torch.no_grad():
                initial_text = self._tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                generate_ids = self._model.generate(input_ids, max_length=128)
                text = self._tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print(initial_text)
                print(text)

        return loss
    
    def validation_step(self, batch, _):
        input_ids = batch["source_ids"]
        attention_mask = batch["source_mask"]
        labels = batch["target_ids"]

        loss, _ = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return loss
