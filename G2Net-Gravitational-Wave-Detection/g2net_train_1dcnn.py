#!/usr/bin/env python3

import torch
import gc
import timm
import wandb
import torch.nn as nn
import numpy as np
import pandas as pd
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

DATA_DIR = "/home/egortrushin/datasets/g2net-gravitational-wave-detection"
OUTPUT_DIR = "./Checkpoints/"

config = {
    "seed": 1,
    "fold_seed": 42,
    "train_folds": [1],
    "base_model": "1dcnn",
    "base_model_classifier": "classifier",
    "classes": 1,
    "precision": 16,
    "train_batch_size": 64,
    "val_batch_size": 64,
    "epochs": 30,
    "num_workers": 4,
    "weight_decay": 1e-6,
    "lr": 3e-3,
    "min_lr": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "t_max": 30,
    "hop_length": [32, 32, 32],
    "bins_per_octave": [6, 12, 24],
    "fmin": 20,
    "fmax": 512,
    "image_size": (128, 128),
    "freq_sup": False,
    "logger": "wandb",
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
                                  name=f'b1_ns 3ch tests f{fold}',
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


class GeM(nn.Module):
    '''
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    '''
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class G2NetClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cnn1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=32),
            GeM(kernel_size=8),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=32),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16),
            GeM(kernel_size=6),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.SiLU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=16),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=16),
            GeM(kernel_size=4),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.SiLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 11, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.SiLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.SiLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

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
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config["t_max"],
            eta_min=self.config["min_lr"],
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
    early_stop_callback = EarlyStopping(monitor="val_score", mode="max", patience=8)
    
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
        #limit_train_batches=0.1, limit_val_batches=0.1,
        logger = None#wandb_logger,
    )

    model = G2NetClassifier(config)
    trainer.fit(model, data_module)
    
    if config["logger"] is not None:
        wandb.finish()

    del model
    gc.collect()
