#!/usr/bin/env python3

import numpy as np
import pandas as pd


def balanced_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    nc = np.bincount(y_true)
    w0, w1 = 1 / (nc[0] / y_true.shape[0]), 1 / (nc[1] / y_true.shape[0])
    balanced_log_loss_score = (
        -1.0 / nc[0] * (np.sum(np.where(y_true == 0, 1, 0) * np.log(1 - y_pred)))
        - 1.0 / nc[1] * (np.sum(np.where(y_true != 0, 1, 0) * np.log(y_pred)))
    ) / 2.0
    return balanced_log_loss_score


train = pd.read_csv(
    "/home/trushin/Kaggle/icr-identify-age-related-conditions/train.csv"
)

MODELS = ["catboost_1748", "lightgbm_1831"]
WEIGHTS = [0.65, 0.35]

train["oof_preds"] = 0.0
for i, model in enumerate(MODELS):
    oof = pd.read_csv(model+"/oof.csv")
    print(f"CV {model} = {balanced_log_loss(train['Class'], oof['prediction']):.4f}")
    train["oof_preds"] += WEIGHTS[i]*oof["prediction"]

print(f"Ensemble CV = {balanced_log_loss(train['Class'], train['oof_preds']):.4f}")