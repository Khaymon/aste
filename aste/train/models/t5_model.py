from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import T5ForConditionalGeneration, T5Tokenizer


@dataclass
class ConditionalGenerationHparams:
    model_name: str
    tokenizer_name: str
    
    train_batch_size: int
    eval_batch_size: int


class T5Model(pl.LightningModule):
    def __init__(self, hparams: ConditionalGenerationHparams):
        super().__init__()
        
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name)
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        
    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels == self.tokenizer.pad_token]
    
    def training_step(self, batch, batch_ids):
        loss = self()