#!/usr/bin/env python3

import pandas as pd
import torch
import gc
import timm
import wandb
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from nnAudio.Spectrogram import CQT1992v2
from scipy import signal
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "/home/woody/bctc/bctc33/g2net"
OUTPUT_DIR = "/home/woody/bctc/bctc33/pytorch/tinygpu_4/Checkpoints/"

config = {
    "seed": 1,
    "fold_seed": 42,
    "train_folds": [0],
    "base_model": "tf_efficientnet_b3_ns",
    "base_model_classifier": "classifier",
    "classes": 1,
    "precision": 32,
    "train_batch_size": 128,
    "val_batch_size": 512,
    "epochs": 30,
    "num_workers": 4,
    "weight_decay": 1e-6,
    "lr": 3e-3,
    "min_lr": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "t_max": 30,
    "hop_length": [8, 8, 8],
    "bins_per_octave": [6, 12, 24],
    "fmin": 20,
    "fmax": 512,
    "image_size": (96, 512),
    "freq_sup": False,
    "logger": None,
    "gradient_clip_val": 1.0e3,
}


def get_path(x, basedir):
    return f"{basedir}/{x[0]}/{x[1]}/{x[2]}/{x}.npy"


def apply_bandpass(x, lf=25, hf=500, order=4, sr=2048):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    return signal.sosfiltfilt(sos, x) / normalization

def get_logger(fold: int):
    return pl_loggers.WandbLogger(project='G2Net',
                                  config=config,
                                  name=f'b1_ns f{fold}',
                                  save_dir="./logs/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.enable()

pl.seed_everything(config["seed"])

df = pd.read_csv(Path(DATA_DIR, "training_labels.csv"))

df["path"] = df.id.apply(get_path, basedir=Path(DATA_DIR, "train"))

Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["fold_seed"])
for n, (train_index, val_index) in enumerate(Fold.split(df, df["target"])):
    df.loc[val_index, "fold"] = int(n)
df["fold"] = df["fold"].astype(int)


class G2NetDataset(Dataset):
    def __init__(self, paths, targets=None):
        self.paths = paths
        self.targets = targets

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waves = np.load(self.paths[idx])
        for i in range(3):
            waves[i] = waves[i]*signal.tukey(4096, 0.2)
            waves[i] = apply_bandpass(waves[i], 35, 500) * 5.0e20
        waves = torch.from_numpy(waves).float()
        if self.targets is not None:
            return waves, torch.tensor(self.targets[idx], dtype=torch.long)
        else:
            return waves


class G2NetDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, config):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.config = config

    def setup(self, stage=None):

        # Create train dataset
        self.train_dataset = G2NetDataset(
            self.train_df.path.values, self.train_df.target.values
        )

        # Create val dataset
        self.val_dataset = G2NetDataset(
            self.val_df.path.values, self.val_df.target.values
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["train_batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["val_batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            pin_memory=False,
        )


class G2NetClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.spec_layer1 = CQT1992v2(
            sr=2048,
            fmin=config["fmin"],
            fmax=config["fmax"],
            hop_length=config["hop_length"][0],
            bins_per_octave=config["bins_per_octave"][0],
            verbose=False,
            trainable=False,
        )

        self.spec_layer2 = CQT1992v2(
            sr=2048,
            fmin=config["fmin"],
            fmax=config["fmax"],
            hop_length=config["hop_length"][1],
            bins_per_octave=config["bins_per_octave"][1],
            verbose=False,
            trainable=False,
        )

        self.spec_layer3 = CQT1992v2(
            sr=2048,
            fmin=config["fmin"],
            fmax=config["fmax"],
            hop_length=config["hop_length"][2],
            bins_per_octave=config["bins_per_octave"][2],
            verbose=False,
            trainable=False,
        )


        self.classifier = timm.create_model(self.hparams.base_model, pretrained=True, in_chans=9)
        n_features = self.classifier._modules[
            self.hparams.base_model_classifier
        ].in_features
        self.classifier._modules[self.hparams.base_model_classifier] = nn.Linear(
            n_features, self.hparams.classes
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):

        z0 = self.spec_layer1(x[:, 0, :])
        z1 = self.spec_layer1(x[:, 1, :])
        z2 = self.spec_layer1(x[:, 2, :])
        x1 = torch.stack((z0, z1, z2), 1)
        x1 = nn.functional.interpolate(x1, size=config["image_size"], mode='bicubic')

        z0 = self.spec_layer2(x[:, 0, :])
        z1 = self.spec_layer2(x[:, 1, :])
        z2 = self.spec_layer2(x[:, 2, :])
        x2 = torch.stack((z0, z1, z2), 1)
        x2 = nn.functional.interpolate(x2, size=config["image_size"], mode='bicubic')

        z0 = self.spec_layer3(x[:, 0, :])
        z1 = self.spec_layer3(x[:, 1, :])
        z2 = self.spec_layer3(x[:, 2, :])
        x3 = torch.stack((z0, z1, z2), 1)
        x3 = nn.functional.interpolate(x3, size=config["image_size"], mode='bicubic')

        x = torch.cat((x1, x2, x3), 1)

        if config["freq_sup"]:
            freq_sup = torch.logspace(np.log10(config["fmin"]/config["fmax"]), 1.0, x.shape[2]).to(device)
            freq_sup = freq_sup[None,None,:,None].repeat(x.shape[0], x.shape[1], 1, 1)
            x = torch.cat((x, freq_sup), 3)
        out = self.classifier(x)
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(1)
        loss = self.loss(logits, labels.float())
        y_true = labels.cpu().numpy()
        y_pred = logits.cpu().detach().numpy()
        score = roc_auc_score(y_true, y_pred)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_score", score, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "train_score": score}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(1)
        loss = self.loss(logits, labels.float())
        y_true = labels.cpu().numpy()
        y_pred = logits.cpu().detach().numpy()
        score = roc_auc_score(y_true, y_pred)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_score", score, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "val_score": score}

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=self.hparams.min_lr,
            last_epoch=-1,
        )
        return {"optimizer": optimizer, "scheduler": scheduler}



for fold in config["train_folds"]:
    print(f"*** fold {fold} ***")

    train_df = df.loc[df.fold != fold]
    val_df = df.loc[df.fold == fold]

    data_module = G2NetDataModule(train_df, val_df, config)

    filename = f"{config['base_model']}-f{fold}-{{val_score:.5f}}-{{val_loss:.4f}}"

    checkpoint_callback = ModelCheckpoint(
        monitor="val_score", dirpath=OUTPUT_DIR, mode="max", filename=filename, save_top_k=2, verbose=1
    )
    early_stop_callback = EarlyStopping(monitor="val_score", mode="max", patience=5)

    if config["logger"] is not None:
        wandb_logger = get_logger(fold)
    else:
        wandb_logger = None

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config["epochs"],
        precision=config["precision"],
        deterministic=True,
        stochastic_weight_avg=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        #val_check_interval = 0.2,
        #limit_train_batches=0.01, limit_val_batches=0.01,
        logger = wandb_logger,
        gradient_clip_val=config["gradient_clip_val"],
        progress_bar_refresh_rate=100,
    )

    model = G2NetClassifier(config)
    trainer.fit(model, data_module)

    if config["logger"] is not None:
        wandb.finish()

    del model
    gc.collect()
