#!/usr/bin/env python3
"""Training script for BirdCLEF 2022."""

import glob
import yaml
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from BirdCLEF_DataModule import ALL_BIRDS, compute_melspec, mono_to_color
from BirdCLEF_Model import BirdCLEFModel


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, clip, AudioParams):
        self.df = df
        self.clip = np.concatenate([clip, clip, clip])
        self.AudioParams = AudioParams

        mean = (0.485, 0.456, 0.406)  # RGB
        std = (0.229, 0.224, 0.225)  # RGB

        self.albu_transforms = {
            "valid": A.Compose(
                [
                    A.Normalize(mean, std),
                ]
            ),
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)

        image = self.clip[self.AudioParams["sr"] * start_seconds : self.AudioParams["sr"] * end_seconds].astype(
            np.float32
        )
        image = np.nan_to_num(image)

        image = compute_melspec(image, self.AudioParams)
        image = mono_to_color(image)
        image = image.astype(np.uint8)

        image = self.albu_transforms["valid"](image=image)["image"].T

        return {
            "image": image,
            "row_id": row_id,
        }


def prediction_for_clip(test_df, clip, models, config, threshold=0.05, threshold_long=None):

    dataset = TestDataset(
        df=test_df,
        clip=clip,
        AudioParams=config["AudioParams"],
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_dict = {}
    all_preds = []
    for data in loader:
        preds = []
        row_id = data["row_id"][0]
        image = data["image"].to(device)

        for i, model in enumerate(models):
            model_preds = np.empty((0, len(ALL_BIRDS)))
            with torch.cuda.amp.autocast():
                output = model(image)
            model_preds = np.concatenate([model_preds, output["clipwise_output"].detach().cpu().numpy()])
            preds.append(model_preds)

        preds = np.array(preds)
        all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=1)
    all_preds = np.mean(all_preds, axis=0)
    events = all_preds >= threshold

    for i, id in enumerate([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]):
        labels = np.argwhere(events[i]).reshape(-1).tolist()
        if len(labels) == 0:
            prediction_dict[str(row_id)] = "nocall"
        else:
            labels_str_list = list(map(lambda x: ALL_BIRDS[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[str(id)] = label_string
    return prediction_dict


def prediction(test_audios, models, config, threshold=0.05, threshold_long=None):

    prediction_dicts = {}
    for audio_path in test_audios:
        clip, _ = sf.read(audio_path, always_2d=True)
        if len(clip.shape) > 1:  # there are (X, 2) arrays
            clip = np.mean(clip, 1)

        print(f"\nMaking predictions for audio {audio_path}")

        seconds = []
        row_ids = []
        for second in range(5, 65, 5):
            row_id = "_".join(audio_path.name.split(".")[:-1]) + f"_{second}"
            seconds.append(second)
            row_ids.append(row_id)
        test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
        prediction_dict = prediction_for_clip(
            test_df,
            clip=clip,
            models=models,
            threshold=threshold,
            threshold_long=threshold_long,
            config=config,
        )
        prediction_dicts.update(prediction_dict)
    return prediction_dicts


CHKPT_PATH = "ckpts/"
DATA_PATH = "/home/egortrushin/datasets/birdclef-2022"

with open(Path(CHKPT_PATH, "config.yaml"), "r") as file_obj:
    config = yaml.safe_load(file_obj)
config["model"]["n_mels"] = config["data_module"]["AudioParams"]["n_mels"]

threshold = 0.02
threshold_long = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_DIR = Path(DATA_PATH, "test_soundscapes")
all_audios = list(AUDIO_DIR.glob("*.ogg"))

sample_submission = pd.read_csv(Path(DATA_PATH, "sample_submission.csv"))

models = []
for path in glob.glob(CHKPT_PATH + "/*.ckpt"):
    print(path)
    model = BirdCLEFModel.load_from_checkpoint(path, config=config["model"])
    model.to(device)
    model.eval()
    models.append(model)


pred_dicts = prediction(
    all_audios,
    models,
    config=config["data_module"],
    threshold=threshold,
    threshold_long=threshold_long,
)

for i in range(len(sample_submission)):
    sample = sample_submission.row_id[i]
    key = sample.split("_")[0] + "_" + sample.split("_")[1] + "_" + sample.split("_")[3]
    target_bird = sample.split("_")[2]
    if key in pred_dicts:
        sample_submission.iat[i, 1] = target_bird in pred_dicts[key]

sample_submission.to_csv("submission.csv", index=False)
