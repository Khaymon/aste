import pytorch_lightning as pl
import torch
import transformers

from aste.train.recipes import TrainRecipe


class BaseModel(pl.LightningModule):
    def __init__(self, recipe: TrainRecipe):
        super().__init__()
        
        self._recipe = recipe
        self._model = getattr(transformers, recipe.model_class_name).from_pretrained(recipe.model_name)
        self._tokenizer = getattr(transformers, recipe.tokenizer_class_name).from_pretrained(recipe.model_name)

        for param_num, param in enumerate(self._model.parameters()):
            if param_num < recipe.freeze:
                param.requires_grad = False
        
    def forward(self, **kwargs):
        raise NotImplementedError
        
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
