#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")

import gc
import glob
import os
import torch
import yaml
import pandas as pd
import pytorch_lightning as pl
from datasets import ContrailsDataset
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pl_module import LightningModule
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

torch.set_float32_matmul_precision("medium")

with open("config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)
pprint(config)

pl.seed_everything(config["seed"])

gc.enable()

contrails = os.path.join(config["data_path"], "contrails/")
train_path = os.path.join(config["data_path"], "train_df.csv")
valid_path = os.path.join(config["data_path"], "valid_df.csv")

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)

train_df["path"] = contrails + train_df["record_id"].astype(str) + ".npy"
valid_df["path"] = contrails + valid_df["record_id"].astype(str) + ".npy"

if config["folds_split"]:
    df = pd.concat([train_df, valid_df]).reset_index()

    Fold = KFold(shuffle=True, **config["folds"])
    for n, (trn_index, val_index) in enumerate(Fold.split(df)):
        df.loc[val_index, "kfold"] = int(n)
    df["kfold"] = df["kfold"].astype(int)

for fold in config["train_folds"]:
    if config["folds_split"]:
        print(f"\n###### Fold {fold}")
        trn_df = df[df.kfold != fold].reset_index(drop=True)
        vld_df = df[df.kfold == fold].reset_index(drop=True)
    else:
        trn_df = train_df
        vld_df = valid_df

    ds_trn = ContrailsDataset(
        trn_df, config["model"]["image_size"], train=True, flips=config["flips"]
    )
    ds_val = ContrailsDataset(vld_df, config["model"]["image_size"], train=False, flips=False)

    dl_trn = DataLoader(
        ds_trn,
        batch_size=config["train_bs"],
        shuffle=True,
        num_workers=config["workers"],
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=config["valid_bs"],
        shuffle=False,
        num_workers=config["workers"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="val_dice",
        dirpath=config["output_dir"],
        mode="max",
        filename=f"model-f{fold}-{{val_dice:.4f}}",
        save_top_k=1,
        verbose=1,
    )

    progress_bar_callback = TQDMProgressBar(refresh_rate=config["progress_bar_refresh_rate"])

    early_stop_callback = EarlyStopping(**config["early_stop"])

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar_callback],
        **config["trainer"],
    )

    config["model"]["scheduler"]["params"]["CosineAnnealingLR"]["T_max"] *= len(dl_trn)
    total_steps = len(dl_trn) * config["trainer"]["max_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    config["model"]["scheduler"]["params"]["cosine_with_hard_restarts_schedule_with_warmup"][
        "num_warmup_steps"
    ] = warmup_steps
    config["model"]["scheduler"]["params"]["cosine_with_hard_restarts_schedule_with_warmup"][
        "num_training_steps"
    ] = total_steps

    if len(glob.glob("*.ckpt")) == 0:
        print("TRAINING MODEL FROM SCRATCH")
        model = LightningModule(config["model"])
    else:
        ckpt_path = glob.glob("*.ckpt")[0]
        print("LOADING_MODEL:", ckpt_path)
        model = LightningModule(config["model"]).load_from_checkpoint(
            ckpt_path, config=config["model"]
        )

    trainer.fit(model, dl_trn, dl_val)

    del (
        ds_trn,
        ds_val,
        dl_trn,
        dl_val,
        model,
        trainer,
        checkpoint_callback,
        progress_bar_callback,
        early_stop_callback,
    )
    torch.cuda.empty_cache()
    gc.collect()
