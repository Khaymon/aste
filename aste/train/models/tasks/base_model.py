import pytorch_lightning as pl
import torch
import transformers

from aste.train.recipes import TrainRecipe
from aste.train.data_providers.data_module import DataModule


class BaseModel(pl.LightningModule):
    def __init__(self, recipe: TrainRecipe):
        super().__init__()
        
        self._recipe = recipe
        self._tokenizer = getattr(transformers, recipe["model"]["hub_tokenizer_name"]).from_pretrained(recipe["model"]["hub_tokenizer_checkpoint"])

        self._model = getattr(transformers, recipe["model"]["hub_model_name"]).from_pretrained(recipe["model"]["hub_model_checkpoint"])

        for param_num, param in enumerate(self._model.parameters()):
            if param_num < recipe["train"]["freeze"]:
                param.requires_grad = False
        
    def forward(self, **kwargs):
        raise NotImplementedError
        
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        train_data_module = DataModule.get_dataloader(self._recipe["model"], self._recipe["dataloaders"]["train"])

        optim = torch.optim.Adam(self.parameters(), lr=3e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_data_module))

        return [optim], [sched]
