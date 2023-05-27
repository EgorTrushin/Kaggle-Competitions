#!/usr/bin/env python3
"""Training script for BirdCLEF 2022."""

import ast
import gc
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from BirdCLEF_DataModule import BirdCLEFDataModule
from BirdCLEF_Model import BirdCLEFModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold


def process_data(data_path):
    """Read and process metadata file."""
    df = pd.read_csv(Path(data_path, "train_metadata.csv"))
    df["new_target"] = df["primary_label"] + " " + df["secondary_labels"].map(lambda x: " ".join(ast.literal_eval(x)))
    df["len_new_target"] = df["new_target"].map(lambda x: len(x.split()))
    df["file_path"] = data_path + "/train_audio/" + df["filename"]
    return df


def create_folds(df, **kwargs):
    """Perform fold splitting."""
    Fold = StratifiedKFold(shuffle=True, **kwargs)
    for n, (trn_index, val_index) in enumerate(Fold.split(df, df["primary_label"])):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)
    return df


if __name__ == "__main__":
    with open("config.yaml", "r") as file_obj:
        config = yaml.safe_load(file_obj)
    config["model"]["n_mels"] = config["data_module"]["AudioParams"]["n_mels"]

    df = process_data(config["data_path"])
    df = create_folds(df, **config["folds"])

    gc.enable()

    pl.seed_everything(config["seed"])

    for fold in config["train_folds"]:
        print(f"\n###### Fold {fold}")

        train_df = df[df.kfold != fold].reset_index(drop=True)
        valid_df = df[df.kfold == fold].reset_index(drop=True)

        data_module = BirdCLEFDataModule(train_df, valid_df, config["data_module"])

        chkpt_callback = ModelCheckpoint(
            filename=f"f{fold}-{{val_score:.5f}}-{{val_loss:.5f}}",
            **config["ckpt_callback"],
        )
        es_callback = EarlyStopping(**config["es_callback"])

        trainer = pl.Trainer(callbacks=[chkpt_callback, es_callback], logger=None, **config["trainer"])

        model = BirdCLEFModel(config["model"])
        trainer.fit(model, data_module)

        del data_module, trainer, model, chkpt_callback, es_callback
        torch.cuda.empty_cache()
        gc.collect()
