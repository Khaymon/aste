import argparse
import pathlib

from aste.data.readers import BankDataReader
from aste.train.data_providers import ModelDatasetModule
from aste.train.models import BaseGenerationModel
from aste.train.recipes import TrainRecipe

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch


def _parse_args():
    parser = argparse.ArgumentParser(
        "Train model",
        "Train model with Lightning",
        add_help=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--recipe", type=pathlib.Path, required=True, help="Path to the train recipe")
    parser.add_argument("--gpus", type=int, nargs="+", required=True, help="Number of GPUs to use for train")

    return parser.parse_args()


def main():
    args = _parse_args()

    recipe = TrainRecipe.from_file(args.recipe)

    train_data = BankDataReader.from_file(recipe.train_path)
    dev_data = BankDataReader.from_file(recipe.dev_path)
    dataset_module = ModelDatasetModule(recipe=recipe, train_data=train_data, dev_data=dev_data)

    model = BaseGenerationModel(recipe)

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=1, monitor="val_loss")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=1,
        mode="min"
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=recipe.epochs,
    )

    trainer.fit(model, dataset_module.train_dataloader(), dataset_module.val_dataloader())


if __name__ == "__main__":
    main()