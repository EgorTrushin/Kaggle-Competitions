#!/usr/bin/env python3

import yaml
import pickle
from pathlib import Path
import gc
from glob import glob

import numpy as np
import polars as pl
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb

from pprint import pprint

import time
from datetime import timedelta


class Pipeline:
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
                df = df.with_columns(pl.col(col).dt.total_days())  # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.7:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df


class Aggregator:
    # Please add or subtract features yourself, be aware that too many features will take up too much space.
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_last + expr_mean

    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_last + expr_mean

    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        # expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]
        return expr_max + expr_last  # +expr_count

    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        return expr_max + expr_last

    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        return expr_max + expr_last

    def get_exprs(df):
        exprs = (
            Aggregator.num_expr(df)
            + Aggregator.date_expr(df)
            + Aggregator.str_expr(df)
            + Aggregator.other_expr(df)
            + Aggregator.count_expr(df)
        )

        return exprs


def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df


def read_files(regex_path, depth=None):
    chunks = []

    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)

    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df


def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = df_base.with_columns(
        month_decision=pl.col("date_decision").dt.month(),
        weekday_decision=pl.col("date_decision").dt.weekday(),
    )
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == "category":
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def reduce_group(grps, df_train):
    use = []
    for g in grps:
        mx = 0
        vx = g[0]
        for gg in g:
            n = df_train[gg].nunique()
            if n > mx:
                mx = n
                vx = gg
        use.append(vx)
    # print("Use these", use)
    return use


def group_columns_by_correlation(matrix, threshold=0.8):
    correlation_matrix = matrix.corr()

    groups = []
    remaining_cols = list(matrix.columns)
    while remaining_cols:
        col = remaining_cols.pop(0)
        group = [col]
        correlated_cols = [col]
        for c in remaining_cols:
            if correlation_matrix.loc[col, c] >= threshold:
                group.append(c)
                correlated_cols.append(c)
        groups.append(group)
        remaining_cols = [c for c in remaining_cols if c not in correlated_cols]

    return groups


def gini_stability(base, score_col="score", w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = (
        base.loc[:, ["WEEK_NUM", "target", score_col]]
        .sort_values("WEEK_NUM")
        .groupby("WEEK_NUM")[["target", score_col]]
        .apply(lambda x: 2 * roc_auc_score(x["target"], x[score_col]) - 1)
        .tolist()
    )

    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std


def calc_log_loss_weight(y_true):
    """Calculates weights for dataset."""
    nc = np.bincount(y_true)
    w0, w1 = 1 / (nc[0] / y_true.shape[0]), 1 / (nc[1] / y_true.shape[0])
    return w0, w1


def train_catboost(x_train, y_train, x_valid, y_valid, categorical_features, cat_params, use_class_weights):
    """catboost training."""
    if use_class_weights:
        train_w0, train_w1 = calc_log_loss_weight(y_train)
        valid_w0, valid_w1 = calc_log_loss_weight(y_valid)

        cat_train = Pool(
            data=x_train,
            label=y_train,
            weight=y_train.map({0: train_w0, 1: train_w1}),
            cat_features=categorical_features,
        )
        cat_valid = Pool(
            data=x_valid,
            label=y_valid,
            weight=y_valid.map({0: valid_w0, 1: valid_w1}),
            cat_features=categorical_features,
        )
    else:
        cat_train = Pool(
            data=x_train,
            label=y_train,
            cat_features=categorical_features,
        )
        cat_valid = Pool(
            data=x_valid,
            label=y_valid,
            cat_features=categorical_features,
        )

    model = CatBoostClassifier(**cat_params)
    model.fit(cat_train, eval_set=[cat_valid], use_best_model=True)

    valid_pred = model.predict_proba(x_valid)[:, 1]

    return model, valid_pred


def train_lightgbm(x_train, y_train, x_valid, y_valid, lgb_params, train_params, use_class_weights):
    """LightGBM training."""
    if use_class_weights:
        train_w0, train_w1 = calc_log_loss_weight(y_train)
        valid_w0, valid_w1 = calc_log_loss_weight(y_valid)

        lgb_train = lgb.Dataset(
            x_train,
            y_train,
            weight=y_train.map({0: train_w0, 1: train_w1}),
        )
        lgb_valid = lgb.Dataset(
            x_valid,
            y_valid,
            weight=y_valid.map({0: valid_w0, 1: valid_w1}),
        )
    else:
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_valid = lgb.Dataset(x_valid, y_valid)

    model = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[
            lgb.early_stopping(train_params["stopping_rounds"]),
            lgb.log_evaluation(train_params["period"]),
        ],
    )

    valid_pred = model.predict(x_valid)

    return model, valid_pred


def train_xgboost(x_train, y_train, x_valid, y_valid, xgb_params, train, use_class_weights):
    """XGBoost training."""
    if use_class_weights:
        train_w0, train_w1 = calc_log_loss_weight(y_train)
        valid_w0, valid_w1 = calc_log_loss_weight(y_valid)
        xgb_train = xgb.DMatrix(
            data=x_train, label=y_train, weight=y_train.map({0: train_w0, 1: train_w1}, enable_categorical=True)
        )
        xgb_valid = xgb.DMatrix(
            data=x_valid, label=y_valid, weight=y_valid.map({0: valid_w0, 1: valid_w1}, enable_categorical=True)
        )
    else:
        xgb_train = xgb.DMatrix(data=x_train, label=y_train, enable_categorical=True)
        xgb_valid = xgb.DMatrix(data=x_valid, label=y_valid, enable_categorical=True)

    model = xgb.train(xgb_params, dtrain=xgb_train, evals=[(xgb_train, "train"), (xgb_valid, "eval")], **train)

    valid_pred = model.predict(xgb.DMatrix(x_valid, enable_categorical=True))
    return model, valid_pred


def run_training(config_path):
    t_start = time.time()

    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    ROOT = Path(config["data_path"])
    TRAIN_DIR = ROOT / "parquet_files" / "train"

    print(config_path)
    pprint(config)

    data_store = {
        "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
        "depth_0": [
            read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
            read_files(TRAIN_DIR / "train_static_0_*.parquet"),
        ],
        "depth_1": [
            read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
            read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
            read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
            read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
            read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
            read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
            read_file(TRAIN_DIR / "train_other_1.parquet", 1),
            read_file(TRAIN_DIR / "train_person_1.parquet", 1),
            read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
            read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
            read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),
            read_file(TRAIN_DIR / "train_applprev_2.parquet", 2),
            read_file(TRAIN_DIR / "train_person_2.parquet", 2),
        ],
    }

    df_train = feature_eng(**data_store)
    print("train data shape:\t", df_train.shape)
    del data_store
    gc.collect()
    df_train = df_train.pipe(Pipeline.filter_cols)
    df_train, cat_cols = to_pandas(df_train)
    df_train = reduce_mem_usage(df_train)
    print("train data shape:\t", df_train.shape)
    nums = df_train.select_dtypes(exclude="category").columns
    nans_df = df_train[nums].isna()
    nans_groups = {}
    for col in nums:
        cur_group = nans_df[col].sum()
        try:
            nans_groups[cur_group].append(col)
        except:
            nans_groups[cur_group] = [col]
    del nans_df
    x = gc.collect()

    uses = []
    for k, v in nans_groups.items():
        if len(v) > 1:
            Vs = nans_groups[k]
            grps = group_columns_by_correlation(df_train[Vs], threshold=0.8)
            use = reduce_group(grps, df_train)
            uses = uses + use
        else:
            uses = uses + v
        # print("####### NAN count =", k)
    # print(len(uses))
    uses = uses + list(df_train.select_dtypes(include="category").columns)
    # print(len(uses))
    df_train = df_train[uses]

    y = df_train["target"]
    weeks = df_train["WEEK_NUM"]
    df_train = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])

    df1 = pd.read_csv("/home/trushin/Kaggle/home-credit/home-credit-models/582_CatBoost_CPU_69240/oof.csv")
    df2 = pd.read_csv("/home/trushin/Kaggle/home-credit/home-credit-models/581_CatBoost_69382/oof.csv")
    df3 = pd.read_csv("/home/trushin/Kaggle/home-credit/home-credit-models/573_LightGBM_69915/oof.csv")
    df4 = pd.read_csv("/home/trushin/Kaggle/home-credit/home-credit-models/576_LightGBM_69476/oof.csv")
    df5 = pd.read_csv("/home/trushin/Kaggle/home-credit/home-credit-models/540_XGBoost_69736/oof.csv")

    df_train["p1"] = df1["prediction"]
    df_train["p2"] = df2["prediction"]
    df_train["p3"] = df3["prediction"]
    df_train["p4"] = df4["prediction"]
    df_train["p5"] = df5["prediction"]

    cv = StratifiedGroupKFold(**config["folds"])

    if config["method"] == "catboost":
        df_train[cat_cols] = df_train[cat_cols].astype(str)

    fitted_models = []
    cv_scores = []

    oof = np.zeros(len(df_train))

    for fold, (idx_train, idx_valid) in enumerate(cv.split(df_train, y, groups=weeks)):
        print(f"\n###### Fold {fold}", flush=True)

        x_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train]
        x_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]

        if config["method"] == "catboost":
            model, y_pred_valid = train_catboost(
                x_train, y_train, x_valid, y_valid, cat_cols, config["model"], config["use_class_weights"]
            )

        if config["method"] == "lightgbm":
            model, y_pred_valid = train_lightgbm(
                x_train, y_train, x_valid, y_valid, config["model"], config["train"], config["use_class_weights"]
            )

        if config["method"] == "xgboost":
            model, y_pred_valid = train_xgboost(
                x_train, y_train, x_valid, y_valid, config["model"], config["train"], config["use_class_weights"]
            )

        fitted_models.append(model)

        oof[idx_valid] = y_pred_valid

        auc_score = roc_auc_score(y_valid, y_pred_valid)
        cv_scores.append(auc_score)

    print("\nAUC scores:")
    for i, score in enumerate(cv_scores):
        print(f"{i:} {score:.5f}")
    print(f"Mean AUC score: {np.mean(np.array(cv_scores)):.5f}")
    print(f"Mean Gini score: {2.*np.mean(np.array(cv_scores))-1.:.5f}")
    print(f"OOF AUC score: {roc_auc_score(y, oof):.5f}")

    with open("models.pkl", "wb") as file_obj:
        pickle.dump(fitted_models, file_obj)

    oof_df = pd.DataFrame(
        {
            "target": y,
            "prediction": oof,
            "WEEK_NUM": weeks,
        }
    )

    gini_score = gini_stability(oof_df, score_col="prediction")
    print(f"Stability score: {gini_score:.5f}")

    oof_df.to_csv("oof.csv", index=False)

    print("\nElapsed time:", timedelta(seconds=time.time() - t_start))

run_training("config.yaml")
