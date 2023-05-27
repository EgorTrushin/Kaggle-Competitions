#!/usr/bin/env python3

import pandas as pd
import torch
import gc
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold

import warnings

warnings.filterwarnings("ignore")

config = {
    "input_path": "/home/woody/bctc/bctc33/ventilator-pressure-prediction/",
    "ckpt_path": "./Checkpoints/",
    "seed": 3,
    "fold_seed": 42,
    "train_folds": [0],
    "precision": 16,
    "batch_size": 256,
    "val_batch_size": 1024,
    "epochs": 1000,
    "num_workers": 2,
    "weight_decay": 1e-6,
    "lr": 5e-4,
    "gradient_clip_val": 1.0e3,
    "lstm_dim": 768,
    "lstm_num_layers": 4,
    "lstm_dropout": 0.20,
    "dense_dim": 512,
    "logit_dim": 512,
    "scheduler_t0": 50,
    "scheduler_eta_min": 1.0e-5,
    "scheduler_tmult": 1,
    "features_config": 6,
    "robust_scaler": True,
    "no_scale": ["id", "kfold", "breath_id", "u_out", "pressure"],
    "progress_bar_refresh_rate": 100,
    "loss": "CustomLoss",
    "HuberLoss_delta": 1.0,
    "CustomLoss_shift": -0.5,
    "cnn_block": False,
    "transformer_block": True,
    "transformer_layers": 1,
    "transformer_heads": 4,
    "transformer_dim_feedforward": 256,
    "transformer_dropout": 0.01,
    "transformer_eps": 1.0e-5,
}


###### Auxiliary functions


def scale_features(df, config):
    scaler = RobustScaler()
    for scaler_target in df:
        if scaler_target not in config["no_scale"]:
            scaler.fit(df.loc[:, [scaler_target]])
            df.loc[:, [scaler_target]] = scaler.transform(df.loc[:, [scaler_target]])


def masked_metric(df, preds):
    y = np.array(df["p"].cpu().detach())
    w = 1 - np.array(df["u_out"].cpu().detach())

    assert y.shape == preds.shape and w.shape == y.shape, (
        y.shape,
        preds.shape,
        w.shape,
    )

    mae = w * np.abs(y - np.array(preds.cpu().detach()))
    mae = mae.sum() / w.sum()

    return mae


class CustomLoss(nn.Module):
    def __init__(self, shift):
        super(CustomLoss, self).__init__()
        self.shift = shift

    def __call__(self, preds, y, u_out):
        w = 1.0 - (u_out + self.shift)
        mae = w * (y - preds).abs()
        mae = mae.sum(-1) / w.sum(-1)
        return mae.mean()


class CustomTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is inspired by the paper "Attention Is All You Need"
    where we have removed the dropouts and reduced the 1024 internal DIMS as well.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    __constants__ = ["batch_first"]
    # batch_first is False here

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5
                ) -> None:
        
        factory_kwargs = {}
        super(CustomTransformerEncoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, **factory_kwargs)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.SELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def add_features(df, config):
    if config["features_config"] == 1:
        df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
        df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
        df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
        df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
        df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
        df["u_in_lag5"] = df.groupby("breath_id")["u_in"].shift(5)
        df["u_in_lag6"] = df.groupby("breath_id")["u_in"].shift(6)
        df["u_in_lag7"] = df.groupby("breath_id")["u_in"].shift(7)
        df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
        df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
        df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
        df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
        df["u_in_diff5"] = df["u_in"] - df["u_in_lag5"]
        df["u_in_diff6"] = df["u_in"] - df["u_in_lag6"]
        df["u_in_diff7"] = df["u_in"] - df["u_in_lag7"]
        df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
        df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
        df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
        df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
        df["R_dev_C"] = df["R"] / df["C"]
        df["R_minus_C"] = df["R"] - df["C"]
        df = df.fillna(0)

    if config["features_config"] == 2:
        df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
        df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
        df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
        df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
        df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
        df["u_in_lag5"] = df.groupby("breath_id")["u_in"].shift(5)
        df["u_in_lag6"] = df.groupby("breath_id")["u_in"].shift(6)
        df["u_in_lag7"] = df.groupby("breath_id")["u_in"].shift(7)
        df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
        df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
        df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
        df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
        df["u_in_diff5"] = df["u_in"] - df["u_in_lag5"]
        df["u_in_diff6"] = df["u_in"] - df["u_in_lag6"]
        df["u_in_diff7"] = df["u_in"] - df["u_in_lag7"]
        df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
        df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
        df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
        df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
        df["R_dev_C"] = df["R"] / df["C"]
        df["R_minus_C"] = df["R"] - df["C"]
        df["breath_id__u_in__diffmax"] = (
            df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
        )
        df["breath_id__u_in__diffmean"] = (
            df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
        )
        df = df.fillna(0)

    if config["features_config"] == 3:
        df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
        df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
        df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
        df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
        df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
        df["u_in_lag5"] = df.groupby("breath_id")["u_in"].shift(5)
        df["u_in_lag6"] = df.groupby("breath_id")["u_in"].shift(6)
        df["u_in_lag7"] = df.groupby("breath_id")["u_in"].shift(7)
        df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
        df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
        df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
        df["u_out_lag4"] = df.groupby("breath_id")["u_out"].shift(4)
        df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
        df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
        df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
        df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
        df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
        df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
        df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
        df["u_out_lag_back4"] = df.groupby("breath_id")["u_out"].shift(-4)
        df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
        df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
        df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
        df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
        df["u_in_diff5"] = df["u_in"] - df["u_in_lag5"]
        df["u_in_diff6"] = df["u_in"] - df["u_in_lag6"]
        df["u_in_diff7"] = df["u_in"] - df["u_in_lag7"]
        df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
        df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
        df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
        df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
        df["rolling_10_mean"] = (
            df.groupby("breath_id")["u_in"]
            .rolling(window=10, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["R_dev_C"] = df["R"] / df["C"]
        df["R_minus_C"] = df["R"] - df["C"]
        df["R_times_C"] = df["R"] * df["C"]
        df["R_plus_C"] = df["R"] + df["C"]
        df["breath_id__u_in__diffmax"] = (
            df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
        )
        df["breath_id__u_in__diffmean"] = (
            df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
        )
        df = df.fillna(0)

    if config["features_config"] == 4:
        df["u_out_scaled"] = df["u_out"]
        df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
        df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
        df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
        df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
        df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
        df["u_in_lag5"] = df.groupby("breath_id")["u_in"].shift(5)
        df["u_in_lag6"] = df.groupby("breath_id")["u_in"].shift(6)
        df["u_in_lag7"] = df.groupby("breath_id")["u_in"].shift(7)
        df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
        df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
        df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
        df["u_out_lag4"] = df.groupby("breath_id")["u_out"].shift(4)
        df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
        df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
        df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
        df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
        df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
        df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
        df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
        df["u_out_lag_back4"] = df.groupby("breath_id")["u_out"].shift(-4)
        df = df.fillna(0)
        df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
        df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
        df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
        df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
        df["u_in_diff5"] = df["u_in"] - df["u_in_lag5"]
        df["u_in_diff6"] = df["u_in"] - df["u_in_lag6"]
        df["u_in_diff7"] = df["u_in"] - df["u_in_lag7"]
        df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
        df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
        df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
        df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
        df["rolling_10_mean"] = (
            df.groupby("breath_id")["u_in"]
            .rolling(window=10, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["R_dev_C"] = df["R"] / df["C"]
        df["R_minus_C"] = df["R"] - df["C"]
        df["R_times_C"] = df["R"] * df["C"]
        df["R_plus_C"] = df["R"] + df["C"]
        df = df.fillna(0)

    if config["features_config"] == 5:
        df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
        df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
        df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
        df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
        df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
        df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
        df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
        df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
        df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
        df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
        df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
        df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
        df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
        df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
        df["u_out_lag4"] = df.groupby("breath_id")["u_out"].shift(4)
        df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
        df["u_out_lag_back4"] = df.groupby("breath_id")["u_out"].shift(-4)
        df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
        df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
        df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
        df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
        df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
        df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
        df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
        df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
        df["rolling_10_mean"] = (
            df.groupby("breath_id")["u_in"]
            .rolling(window=10, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["R_dev_C"] = df["R"] / df["C"]
        df["R_minus_C"] = df["R"] - df["C"]
        df = df.fillna(0)

    if config["features_config"] == 6:
        df["u_out_scaled"] = df["u_out"]
        df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
        df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
        df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
        df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
        df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
        df["u_in_lag5"] = df.groupby("breath_id")["u_in"].shift(5)
        df["u_in_lag6"] = df.groupby("breath_id")["u_in"].shift(6)
        df["u_in_lag7"] = df.groupby("breath_id")["u_in"].shift(7)
        df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
        df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
        df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
        df["u_out_lag4"] = df.groupby("breath_id")["u_out"].shift(4)
        df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
        df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
        df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
        df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
        df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
        df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
        df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
        df["u_out_lag_back4"] = df.groupby("breath_id")["u_out"].shift(-4)
        df = df.fillna(0)
        df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
        df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
        df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
        df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
        df["u_in_diff5"] = df["u_in"] - df["u_in_lag5"]
        df["u_in_diff6"] = df["u_in"] - df["u_in_lag6"]
        df["u_in_diff7"] = df["u_in"] - df["u_in_lag7"]
        df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
        df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
        df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
        df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
        df["rolling_10_mean"] = (
            df.groupby("breath_id")["u_in"]
            .rolling(window=10, min_periods=1)
            .mean() 
            .reset_index(level=0, drop=True)
        )       
        df["R_dev_C"] = df["R"] / df["C"] 
        df["R_minus_C"] = df["R"] - df["C"] 
        df["R_times_C"] = df["R"] * df["C"] 
        df["R_plus_C"] = df["R"] + df["C"] 
        df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
        #df['ewm_u_in_mean'] = (df\
        #                       .groupby('breath_id')['u_in']\
        #                       .ewm(halflife=9)\
        #                       .mean()\
        #                       .reset_index(level=0,drop=True))
        df[["15_in_sum","15_in_min","15_in_max","15_in_mean"]] = (df\
                                                              .groupby('breath_id')['u_in']\
                                                              .rolling(window=15,min_periods=1)\
                                                              .agg({"15_in_sum":"sum",
                                                                    "15_in_min":"min",
                                                                    "15_in_max":"max",
                                                                    "15_in_mean":"mean"})\
                                                               .reset_index(level=0,drop=True))
        df['u_in_lagback_diff1'] = df['u_in'] - df['u_in_lag_back1']
        df['u_out_lagback_diff1'] = df['u_out'] - df['u_out_lag_back1']
        df['u_in_lagback_diff2'] = df['u_in'] - df['u_in_lag_back2']
        df['u_out_lagback_diff2'] = df['u_out'] - df['u_out_lag_back2']
        df = df.fillna(0)


    return df


###### Dataset


class GBVPPDataset(Dataset):
    def __init__(self, df):
        if "pressure" not in df.columns:
            df["pressure"] = 0

        self.df = df.groupby("breath_id").agg(list).reset_index()

        self.prepare_data()

    def __len__(self):
        return self.df.shape[0]

    def prepare_data(self):
        self.pressures = np.array(self.df["pressure"].values.tolist())
        self.u_outs = np.array(self.df["u_out"].values.tolist())

        if config["features_config"] == 0:
            self.inputs = np.concatenate(
                [
                    np.array(self.df["R"].values.tolist())[:, None],
                    np.array(self.df["C"].values.tolist())[:, None],
                    np.array(self.df["u_in"].values.tolist())[:, None],
                ],
                1,
            ).transpose(0, 2, 1)

        if config["features_config"] == 1:
            self.inputs = np.concatenate(
                [
                    np.array(self.df["R"].values.tolist())[:, None],
                    np.array(self.df["C"].values.tolist())[:, None],
                    np.array(self.df["u_in"].values.tolist())[:, None],
                    np.array(self.df["u_in_cumsum"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag5"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag6"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag7"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff4"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff5"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff6"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff7"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["R_dev_C"].values.tolist())[:, None],
                    np.array(self.df["R_minus_C"].values.tolist())[:, None],
                ],
                1,
            ).transpose(0, 2, 1)

        if config["features_config"] == 2:
            self.inputs = np.concatenate(
                [
                    np.array(self.df["R"].values.tolist())[:, None],
                    np.array(self.df["C"].values.tolist())[:, None],
                    np.array(self.df["u_in"].values.tolist())[:, None],
                    np.array(self.df["u_in_cumsum"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag5"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag6"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag7"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff4"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff5"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff6"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff7"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["R_dev_C"].values.tolist())[:, None],
                    np.array(self.df["R_minus_C"].values.tolist())[:, None],
                    np.array(self.df["breath_id__u_in__diffmax"].values.tolist())[
                        :, None
                    ],
                    np.array(self.df["breath_id__u_in__diffmean"].values.tolist())[
                        :, None
                    ],
                ],
                1,
            ).transpose(0, 2, 1)

        if config["features_config"] == 3:
            self.inputs = np.concatenate(
                [
                    np.array(self.df["R"].values.tolist())[:, None],
                    np.array(self.df["C"].values.tolist())[:, None],
                    np.array(self.df["u_in"].values.tolist())[:, None],
                    np.array(self.df["u_in_cumsum"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag5"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag6"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag7"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff4"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff5"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff6"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff7"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["R_dev_C"].values.tolist())[:, None],
                    np.array(self.df["R_minus_C"].values.tolist())[:, None],
                    np.array(self.df["breath_id__u_in__diffmax"].values.tolist())[
                        :, None
                    ],
                    np.array(self.df["breath_id__u_in__diffmean"].values.tolist())[
                        :, None
                    ],
                    np.array(self.df["u_out_lag1"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff4"].values.tolist())[:, None],
                ],
                1,
            ).transpose(0, 2, 1)

        if config["features_config"] == 4:
            self.inputs = np.concatenate(
                [
                    np.array(self.df["R"].values.tolist())[:, None],
                    np.array(self.df["C"].values.tolist())[:, None],
                    np.array(self.df["R_dev_C"].values.tolist())[:, None],
                    np.array(self.df["R_minus_C"].values.tolist())[:, None],
                    np.array(self.df["R_plus_C"].values.tolist())[:, None],
                    np.array(self.df["R_times_C"].values.tolist())[:, None],
                    np.array(self.df["u_in"].values.tolist())[:, None],
                    np.array(self.df["u_in_cumsum"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag5"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag6"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag7"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff4"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff5"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff6"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff7"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff4"].values.tolist())[:, None],
                    np.array(self.df["rolling_10_mean"].values.tolist())[:, None],
                    np.array(self.df["u_out_scaled"].values.tolist())[:, None],
                ],
                1,
            ).transpose(0, 2, 1)

        if config["features_config"] == 5:
            self.inputs = np.concatenate(
                [
                    np.array(self.df["R"].values.tolist())[:, None],
                    np.array(self.df["C"].values.tolist())[:, None],
                    np.array(self.df["R_dev_C"].values.tolist())[:, None],
                    np.array(self.df["R_minus_C"].values.tolist())[:, None],
                    np.array(self.df["u_in"].values.tolist())[:, None],
                    np.array(self.df["u_in_cumsum"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff4"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff4"].values.tolist())[:, None],
                    np.array(self.df["rolling_10_mean"].values.tolist())[:, None],
                    np.array(self.df["u_out_scaled"].values.tolist())[:, None],
                ],
                1,
            ).transpose(0, 2, 1)


        if config["features_config"] == 6:
            self.inputs = np.concatenate(
                [
                    np.array(self.df["R"].values.tolist())[:, None],
                    np.array(self.df["C"].values.tolist())[:, None],
                    np.array(self.df["R_dev_C"].values.tolist())[:, None],
                    np.array(self.df["R_minus_C"].values.tolist())[:, None],
                    np.array(self.df["R_plus_C"].values.tolist())[:, None],
                    np.array(self.df["R_times_C"].values.tolist())[:, None],
                    np.array(self.df["u_in"].values.tolist())[:, None],
                    np.array(self.df["u_in_cumsum"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag5"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag6"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag7"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag2"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag3"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag4"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_in_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back1"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back2"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back3"].values.tolist())[:, None],
                    np.array(self.df["u_out_lag_back4"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff4"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff5"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff6"].values.tolist())[:, None],
                    np.array(self.df["u_in_diff7"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff3"].values.tolist())[:, None],
                    np.array(self.df["u_out_diff4"].values.tolist())[:, None],
                    np.array(self.df["rolling_10_mean"].values.tolist())[:, None],
                    np.array(self.df["u_out_scaled"].values.tolist())[:, None],
                    np.array(self.df["time_step_diff"].values.tolist())[:, None],
                    #np.array(self.df["ewm_u_in_mean"].values.tolist())[:, None],
                    np.array(self.df["u_in_lagback_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_out_lagback_diff1"].values.tolist())[:, None],
                    np.array(self.df["u_in_lagback_diff2"].values.tolist())[:, None],
                    np.array(self.df["u_out_lagback_diff2"].values.tolist())[:, None],
                    np.array(self.df["15_in_sum"].values.tolist())[:, None],
                    np.array(self.df["15_in_min"].values.tolist())[:, None],
                    np.array(self.df["15_in_max"].values.tolist())[:, None],
                    np.array(self.df["15_in_mean"].values.tolist())[:, None],
                ],
                1,
            ).transpose(0, 2, 1)


    def __getitem__(self, idx):
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data


###### DataModule


class GBVPPDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, config):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = GBVPPDataset(self.train_df)
        self.val_dataset = GBVPPDataset(self.val_df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
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


###### PL Module


class GBVPPModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        if config["features_config"] == 0:
            input_dim = 3
        elif config["features_config"] == 1:
            input_dim = 24
        elif config["features_config"] == 2:
            input_dim = 26
        elif config["features_config"] == 3:
            input_dim = 38
        elif config["features_config"] == 4:
            input_dim = 39
        elif config["features_config"] == 5:
            input_dim = 31
        elif config["features_config"] == 6:
            input_dim = 48
        dense_dim = config["dense_dim"]
        logit_dim = config["logit_dim"]
        num_classes = 1
        self.config = config

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim // 2),
            nn.PReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.PReLU(),
        )

        if config["cnn_block"]:
            self.cnn1 = nn.Conv1d(
                dense_dim, config["lstm_dim"], kernel_size=2, padding=1
            )
            self.cnn2 = nn.Conv1d(
                config["lstm_dim"], dense_dim, kernel_size=2, padding=0
            )

        self.lstm = nn.LSTM(
            dense_dim,
            config["lstm_dim"],
            batch_first=True,
            num_layers=config["lstm_num_layers"],
            dropout=config["lstm_dropout"],
            bidirectional=True,
        )

        if config["transformer_block"]:
            transformer_encoder_layer = CustomTransformerEncoderLayer(
                d_model=config["lstm_dim"] * 2,
                nhead=config["transformer_heads"],
                dim_feedforward=config["transformer_dim_feedforward"],
                dropout=config["transformer_dropout"],
                layer_norm_eps=config["transformer_eps"],
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer=transformer_encoder_layer,
                num_layers=config["transformer_layers"],
            )

        self.logits = nn.Sequential(
            nn.Linear(config["lstm_dim"] * 2, logit_dim),
            nn.PReLU(),
            nn.Linear(logit_dim, num_classes),
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                print(f"init {m}")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
        if config["loss"] == "L1Loss":
            self.loss = nn.L1Loss()
        elif config["loss"] == "HuberLoss":
            self.loss = nn.HuberLoss(delta=config["HuberLoss_delta"])
        elif config["loss"] == "CustomLoss":
            self.loss = CustomLoss(shift=config["CustomLoss_shift"])

    def forward(self, x):
        x = self.mlp(x)
        if self.config["cnn_block"]:
            x = x.permute(0, 2, 1)
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        if config["transformer_block"]:
            x = x.transpose(1, 0)
            self.transformer(x, mask=None, src_key_padding_mask=None)
            x = x.transpose(1, 0)
        pred = self.logits(x)
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch["input"]).squeeze(-1)
        loss = self.loss(pred, batch["p"], batch["u_out"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pred = self(batch["input"]).squeeze(-1)
        loss = self.loss(pred, batch["p"], batch["u_out"])
        y_true = batch
        y_pred = pred
        score = masked_metric(y_true, y_pred)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_score", score, on_step=False, on_epoch=True, prog_bar=True)

        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "val_score": score}

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["scheduler_t0"],
            T_mult=config["scheduler_tmult"],
            eta_min=config["scheduler_eta_min"],
            last_epoch=-1,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


###### Run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.enable()

pl.seed_everything(config["seed"])

# Data processing

df = pd.read_csv(Path(config["input_path"], "train.csv"))

print("\nOriginal Dataframe")
print(df.head())

if config["features_config"] == 0:
    print("Additional features are not used")
else:
    df = add_features(df, config)

if config["robust_scaler"]:
    print("\nRobustScaler is used")
    scale_features(df, config)

# Fold splitting

df["fold"] = -1
y = df.pressure.values
kf = GroupKFold(n_splits=20)
for f, (t_, v_) in enumerate(kf.split(X=df, y=y, groups=df.breath_id.values)):
    df.loc[v_, "fold"] = f

print("\nProcessed Dataframe")
print(df.head())

# Training

for fold in config["train_folds"]:
    print(f"====== fold {fold} ======")

    train_df = df.loc[df.fold != fold]
    val_df = df.loc[df.fold == fold]

    data_module = GBVPPDataModule(train_df, val_df, config)

    filename = f"model-f{fold}-{{val_score:.4f}}-{{val_loss:.4f}}"

    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        dirpath=config["ckpt_path"],
        mode="min",
        filename=filename,
        save_top_k=2,
        verbose=1,
    )

    # early_stop_callback = EarlyStopping(monitor="val_score", mode="min", patience=101)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config["epochs"],
        precision=config["precision"],
        deterministic=True,
        stochastic_weight_avg=False,
        callbacks=[checkpoint_callback],
        logger=None,
        progress_bar_refresh_rate=config["progress_bar_refresh_rate"],
    )

    model = GBVPPModule(config)
    trainer.fit(model, data_module)

    del model
    gc.collect()
