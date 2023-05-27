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
from clrp_utils import create_folds, format_time, oof_predictions


os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda")
NUM_FOLDS = 5
TRAIN_CSV = "~/datasets/commonlitreadabilityprize/train.csv"


t0 = time.time()

df = pd.read_csv(TRAIN_CSV)
df = create_folds(df, random_state=1325)

model_cfg = {
    "model": "roberta-base",
    "weights_dir": "roberta_base_0.47636_0.476/",
    "tokenizer": "roberta-base",
    "max_len": 256,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
}
print(model_cfg)
res_rl = oof_predictions(df, model_cfg, DEVICE)

oof = np.zeros(len(df))
for fold in res_rl:
    oof[res_rl[fold]["val_index"]] += res_rl[fold]["preds"]

with open("oof_predictions.pkl", "wb") as file_obj:
    pickle.dump(oof, file_obj)

cv_score = np.sqrt(mean_squared_error(df.target.values, oof))
print("CV score = {:.5f}".format(cv_score))


print("Total time: {:}".format(format_time(time.time() - t0)))
