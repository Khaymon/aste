import argparse
import pathlib

import toml

from aste.train.data_providers import DataModule
from aste.train.models.model import ASTEModel

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

    return parser.parse_args()


def main():
    args = _parse_args()

    recipe = toml.load(args.recipe)
    model_class = ASTEModel.get_model(recipe["model"]["aste_model_name"])

    if recipe.get("checkpoint"):
        model = model_class.load_from_checkpoint(recipe["checkpoint"], recipe=recipe)
    else:
        model = model_class(recipe)

    checkpoint_callback = ModelCheckpoint(
        **recipe["train"]["callbacks"]["checkpoint"]
    )
    early_stop_callback = EarlyStopping(
        **recipe["train"]["callbacks"]["early_stopping"]
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        **recipe["train"]["trainer"],
    )

    train_dataloader = DataModule.get_dataloader(recipe["model"], recipe["dataloaders"]["train"])
    dev_dataloader = DataModule.get_dataloader(recipe["model"], recipe["dataloaders"]["dev"])

    trainer.fit(model, train_dataloader, dev_dataloader)


if __name__ == "__main__":
    main()