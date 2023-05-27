#!/usr/bin/env python3

import gc
import pandas as pd
from argparse import Namespace
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from datamodule import ToxicDataModule
from model import ToxicModule
import warnings

warnings.filterwarnings("ignore")


config = Namespace(
    fold_seed = 2,
    seed = 5,
    n_folds = 10,
    train_folds = [0],
    #data_path = '/home/egortrushin/datasets/jigsaw-toxic-severity-rating/',
    data_path = '/home/woody/bctc/bctc33/data/jigsaw-toxic-severity-rating/',
    ckpt_path = 'Checkpoints',

    trainer = Namespace(
        precision = 16,
        max_epochs = 10,
        gpus = -1 if torch.cuda.is_available() else 0,
        logger = None,
        gradient_clip_val = 1.0e3,
        progress_bar_refresh_rate = 100,
        val_check_interval = 0.01,
    ),

    model = Namespace(
        loss_margin = 0.1,

        # model layers
        #model_path = 'distilroberta-base',
        model_path = '/home/woody/bctc/bctc33/data/distilroberta-base',
        dropout = 0.2,
        num_classes = 1,
        hidden_size = 512,
        hidden_dropout_prob = 0.1,
        attention_probs_dropout_prob = 0.1,
        num_hidden_layers = 6,

        # optimizer and scheduler
        warmup_ratio = 0.4,
        lr = 2.0e-5,
        weight_decay = 1.0e-2
    ),

    data = Namespace(
        # tokenizer
        #tokenizer_path = 'distilroberta-base',
        tokenizer_path = '/home/woody/bctc/bctc33/data/distilroberta-base',
        max_length = 320,

        # dataloader
        train_batch_size = 64,
        val_batch_size = 512,
        predict_batch_size = 512,
        num_workers = 4,
    ),
)


def get_score(df):
    score = len(df[df['less_toxic_pred'] < df['more_toxic_pred']]) / len(df)
    return score


def main():
    df = pd.read_csv(Path(config.data_path, 'validation_data.csv'))
    predict_df = pd.read_csv(Path(config.data_path, 'pl6.csv'))
    predict_df = predict_df.sample(frac=0.06, random_state=config.seed)

    pl.seed_everything(config.seed)

    gc.enable()

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.fold_seed)
    df['fold'] = -1

    for fold, (_, val_idxs) in enumerate(skf.split(X=df, y=df['worker'])):
        df.loc[val_idxs , 'fold'] = fold


    for fold in config.train_folds:
        print(f"{'#'*5} FOLD: {fold} {'#'*5}")

        dm = ToxicDataModule(df, predict_df, fold=fold, **vars(config.data))
        model = ToxicModule(**vars(config.model))

        filename = f"model-f{fold}-{{val_acc:.5f}}-{{val_loss:.5f}}"
        checkpoint_callback = ModelCheckpoint(
            save_weights_only=True, monitor="val_acc", dirpath=config.ckpt_path, mode="max", filename=filename, verbose=1
        )

        trainer = Trainer(callbacks=[checkpoint_callback], **vars(config.trainer))

        trainer.fit(model, datamodule=dm)

        del trainer, dm, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
