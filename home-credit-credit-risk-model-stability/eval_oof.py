#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def gini_stability(base, score_col="score", w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = base.loc[:, ["WEEK_NUM", "target", score_col]]\
        .sort_values("WEEK_NUM")\
        .groupby("WEEK_NUM")[["target", score_col]]\
        .apply(lambda x: 2*roc_auc_score(x["target"], x[score_col])-1).tolist()
    
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std

def eval_model(model_path, verbose=True):
    result = pd.read_csv(os.path.join(model_path, "oof.csv"))
    if verbose:
        print(model_path)
        score = roc_auc_score(result["target"], result["prediction"])
        print(f"AUC: {score:.5f}")
        score = gini_stability(result, score_col="prediction")
        print(f"Gini: {score:.5f}")
    return result

models = ["home-credit-models/582_CatBoost_CPU_69240",\
          "home-credit-models/581_CatBoost_69382",\
          "home-credit-models/573_LightGBM_69915",\
          "home-credit-models/576_LightGBM_69476",\
          "home-credit-models/540_XGBoost_69736"]

weights = [0.2, 0.2, 0.2, 0.2, 0.2]

predictions = []
for i, model in enumerate(models):
    result = eval_model(model)
    predictions.append(weights[i]*result["prediction"])

print("\nEnsemble")
predictions = np.mean(np.array(predictions), axis=0)
result["prediction"] = predictions

score = roc_auc_score(result["target"], result["prediction"])
print(f"AUC: {score:.5f}")
score = gini_stability(result, score_col="prediction")
print(f"Gini: {score:.5f}")
