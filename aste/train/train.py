import argparse
import pathlib

from aste.train.data_providers import DataModule
from aste.train.models.model import ASTEModel
from aste.train.recipes import TrainRecipe

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def _parse_args():
    parser = argparse.ArgumentParser(
        "Train model",
        "Train model with Lightning",
        add_help=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--recipe", type=pathlib.Path, required=True, help="Path to the train recipe")
    parser.add_argument("--gpus", type=int, nargs="+", required=True, help="Number of GPUs to use for train")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=False, help="Path to the checkpoint")

    return parser.parse_args()


def main():
    args = _parse_args()

    recipe = TrainRecipe.from_file(args.recipe)
    model_class = ASTEModel.get_model(recipe.aste_model_class_name)

    if args.checkpoint:
        model_class.load_from_checkpoint(args.checkpoint, recipe=recipe)
    else:
        model = model_class(recipe)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1,
        monitor="val_loss",
        save_weights_only=True,
    )
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

    train_dataloader = DataModule.get_dataloader(recipe.train_dataloader_recipe)
    dev_data = DataModule.get_dataloader(recipe.dev_dataloader_recipe)

    trainer.fit(model, train_dataloader, dev_data)


if __name__ == "__main__":
    main()