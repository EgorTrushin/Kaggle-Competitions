#!/usr/bin/env python3
import os
import time
import torch
import pickle
import pandas as pd
import numpy as np
import itertools
from transformers import (
    AutoModelForSequenceClassification,
)
from sklearn.metrics import mean_squared_error
from pathlib import Path
from clrp_utils import create_folds, format_time, oof_predictions, get_oof


os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda")
NUM_FOLDS = 5
TRAIN_CSV = "~/datasets/commonlitreadabilityprize/train.csv"


t0 = time.time()

df = pd.read_csv(TRAIN_CSV)
df = create_folds(df, random_state=1325)


#####################################################################
model_cfg = {
    "model": "roberta-large",
    "weights_dir": "roberta_large_0.48146_0.459/",
    "tokenizer": "roberta-large",
    "max_len": 256,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.1,
}

print(model_cfg)

if os.path.exists(Path(model_cfg["weights_dir"], "oof_predictions.pkl")):
    oof = pickle.load(open(Path(model_cfg["weights_dir"], "oof_predictions.pkl"), "rb"))
else:
    res = oof_predictions(df, model_cfg, DEVICE)
    oof = get_oof(df, res)

oof_rl_1 = oof


#####################################################################
model_cfg = {
    "model": "roberta-large",
    "weights_dir": "roberta_large_0.47804_0.464/",
    "tokenizer": "roberta-large",
    "max_len": 256,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.1,
}

print(model_cfg)

if os.path.exists(Path(model_cfg["weights_dir"], "oof_predictions.pkl")):
    oof = pickle.load(open(Path(model_cfg["weights_dir"], "oof_predictions.pkl"), "rb"))
else:
    res = oof_predictions(df, model_cfg, DEVICE)
    oof = get_oof(df, res)

oof_rl_2 = oof


#####################################################################
model_cfg = {
    "model": "roberta-large",
    "weights_dir": "roberta_large_0.48293_0.465/",
    "tokenizer": "roberta-large",
    "max_len": 256,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.1,
}

print(model_cfg)

if os.path.exists(Path(model_cfg["weights_dir"], "oof_predictions.pkl")):
    oof = pickle.load(open(Path(model_cfg["weights_dir"], "oof_predictions.pkl"), "rb"))
else:
    res = oof_predictions(df, model_cfg, DEVICE)
    oof = get_oof(df, res)

oof_rl_3 = oof



#####################################################################
model_cfg = {
    "model": "roberta-base",
    "weights_dir": "roberta_base_0.47766/",
    "tokenizer": "roberta-base",
    "max_len": 256,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.1,
}

print(model_cfg)

if os.path.exists(Path(model_cfg["weights_dir"], "oof_predictions.pkl")):
    oof = pickle.load(open(Path(model_cfg["weights_dir"], "oof_predictions.pkl"), "rb"))
else:
    res = oof_predictions(df, model_cfg, DEVICE)
    oof = get_oof(df, res)

oof_rb_1 = oof


#####################################################################
model_cfg = {
    "model": "roberta-base",
    "weights_dir": "roberta_base_0.47636_0.476/",
    "tokenizer": "roberta-base",
    "max_len": 256,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.1,
}

print(model_cfg)

if os.path.exists(Path(model_cfg["weights_dir"], "oof_predictions.pkl")):
    oof = pickle.load(open(Path(model_cfg["weights_dir"], "oof_predictions.pkl"), "rb"))
else:
    res = oof_predictions(df, model_cfg, DEVICE)
    oof = get_oof(df, res)

oof_rb_2 = oof


#####################################################################
cv_score = np.sqrt(mean_squared_error(df.target.values, 0.4*oof_rl_1+0.3*oof_rl_2+0.15*oof_rb_1+0.15*oof_rb_2))
print("CV score = {:.5f}".format(cv_score))

print("Total time: {:}".format(format_time(time.time() - t0)))
