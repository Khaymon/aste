import torch

from aste.train.recipes import TrainRecipe
from .base_model import BaseModel


class BaseDiscriminatorModel(BaseModel):
    def __init__(self, recipe: TrainRecipe):
        super().__init__(recipe)

        self._fc = torch.nn.Linear(in_features=1024, out_features=2)

    def forward(self, input_ids, mask):
        logits = self._model(
            input_ids=input_ids,
            attention_mask=mask,
        )

        return self._fc(logits.last_hidden_state[:, 0])
        
    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, logger=True)

        if (batch_idx + 1) % 50 == 0:
            with torch.no_grad():
                texts = self._tokenizer.batch_decode(
                    batch["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[:5]
                logits = self.forward(batch["input_ids"], batch["attention_mask"])[:5]
                targets = batch["labels"][:5]
                print(texts)
                print(logits)
                print(targets)

        return loss
    
    def validation_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])

        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return loss
