#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
from clrp_utils import make_predictions

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BATCH_SIZE = 16
NUM_FOLDS = 5
DEVICE = torch.device("cuda")
TEST_CSV = "~/datasets/commonlitreadabilityprize/test.csv"
SAMPLE_CSV = "~/datasets/commonlitreadabilityprize/sample_submission.csv"
model_cfg = {
    "model": "distilroberta-base",
    "weights_dir": "models/distilroberta/",
    "tokenizer": "distilroberta-base",
    "max_len": 256,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
}

predictions = make_predictions(
    model_cfg, TEST_CSV, DEVICE, num_folds=NUM_FOLDS, batch_size=BATCH_SIZE
)

submission = pd.read_csv(SAMPLE_CSV)
submission["target"] = np.asarray(predictions, dtype=np.float32)
submission.to_csv("submission.csv", index=False)

print(submission)
