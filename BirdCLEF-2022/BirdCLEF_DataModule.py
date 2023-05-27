#!/usr/bin/env python3

import random

import albumentations as A
import colorednoise as cn
import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import DataLoader

SCORED_BIRDS = [
    "akiapo",
    "aniani",
    "apapan",
    "barpet",
    "crehon",
    "elepai",
    "ercfra",
    "hawama",
    "hawcre",
    "hawgoo",
    "hawhaw",
    "hawpet1",
    "houfin",
    "iiwi",
    "jabwar",
    "maupar",
    "omao",
    "puaioh",
    "skylar",
    "warwhe1",
    "yefcan",
]

ALL_BIRDS = [
    "afrsil1",
    "akekee",
    "akepa1",
    "akiapo",
    "akikik",
    "amewig",
    "aniani",
    "apapan",
    "arcter",
    "barpet",
    "bcnher",
    "belkin1",
    "bkbplo",
    "bknsti",
    "bkwpet",
    "blkfra",
    "blknod",
    "bongul",
    "brant",
    "brnboo",
    "brnnod",
    "brnowl",
    "brtcur",
    "bubsan",
    "buffle",
    "bulpet",
    "burpar",
    "buwtea",
    "cacgoo1",
    "calqua",
    "cangoo",
    "canvas",
    "caster1",
    "categr",
    "chbsan",
    "chemun",
    "chukar",
    "cintea",
    "comgal1",
    "commyn",
    "compea",
    "comsan",
    "comwax",
    "coopet",
    "crehon",
    "dunlin",
    "elepai",
    "ercfra",
    "eurwig",
    "fragul",
    "gadwal",
    "gamqua",
    "glwgul",
    "gnwtea",
    "golphe",
    "grbher3",
    "grefri",
    "gresca",
    "gryfra",
    "gwfgoo",
    "hawama",
    "hawcoo",
    "hawcre",
    "hawgoo",
    "hawhaw",
    "hawpet1",
    "hoomer",
    "houfin",
    "houspa",
    "hudgod",
    "iiwi",
    "incter1",
    "jabwar",
    "japqua",
    "kalphe",
    "kauama",
    "laugul",
    "layalb",
    "lcspet",
    "leasan",
    "leater1",
    "lessca",
    "lesyel",
    "lobdow",
    "lotjae",
    "madpet",
    "magpet1",
    "mallar3",
    "masboo",
    "mauala",
    "maupar",
    "merlin",
    "mitpar",
    "moudov",
    "norcar",
    "norhar2",
    "normoc",
    "norpin",
    "norsho",
    "nutman",
    "oahama",
    "omao",
    "osprey",
    "pagplo",
    "palila",
    "parjae",
    "pecsan",
    "peflov",
    "perfal",
    "pibgre",
    "pomjae",
    "puaioh",
    "reccar",
    "redava",
    "redjun",
    "redpha1",
    "refboo",
    "rempar",
    "rettro",
    "ribgul",
    "rinduc",
    "rinphe",
    "rocpig",
    "rorpar",
    "rudtur",
    "ruff",
    "saffin",
    "sander",
    "semplo",
    "sheowl",
    "shtsan",
    "skylar",
    "snogoo",
    "sooshe",
    "sooter1",
    "sopsku1",
    "sora",
    "spodov",
    "sposan",
    "towsol",
    "wantat1",
    "warwhe1",
    "wesmea",
    "wessan",
    "wetshe",
    "whfibi",
    "whiter",
    "whttro",
    "wiltur",
    "yebcar",
    "yefcan",
    "zebdov",
]


ALL_BIRDS_IDS = {k: v for v, k in enumerate(ALL_BIRDS)}


SCORED_MASK = list()
for bird in SCORED_BIRDS:
    SCORED_MASK.append(ALL_BIRDS_IDS[bird])


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class OneOf(Compose):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data


class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class NewNormalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise**2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise**2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5):
        super().__init__(always_apply, p)
        self.max_range = max_range

    def apply(self, y: np.ndarray, sr, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1):
        super().__init__(always_apply, p)
        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs


def compute_melspec(y, params):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = librosa.feature.melspectrogram(y=y, **params)

    melspec = librosa.power_to_db(melspec).astype(np.float32)

    return melspec


def crop_or_pad(y, length, sr, train=True, probs=None):
    """
    Crops an array to a chosen length
    Arguments:
        y {1D np array} -- Array to crop
        length {int} -- Length of the crop
        sr {int} -- Sampling rate
    Keyword Arguments:
        train {bool} -- Whether we are at train time. If so, crop randomly,
                        else return the beginning of y (default: {True})
        probs {None or numpy array} -- Probabilities to use to chose where to crop (default: {None})
    Returns:
        1D np array -- Cropped array
    """
    if len(y) <= length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    else:
        if not train:
            start = 0
        elif probs is None:
            start = np.random.randint(len(y) - length)
        else:
            start = np.random.choice(np.arange(len(probs)), p=probs) + np.random.random()
            start = int(sr * (start))

        y = y[start : start + length]

    return y.astype(np.float32)


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Converts a one channel array to a 3 channel one in [0, 255]
    Arguments:
        X {numpy array [H x W]} -- 2D array to convert
    Keyword Arguments:
        eps {float} -- To avoid dividing by 0 (default: {1e-6})
        mean {None or np array} -- Mean for normalization (default: {None})
        std {None or np array} -- Std for normalization (default: {None})
    Returns:
        numpy array [3 x H x W] -- RGB numpy array
    """
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


class AllBirdsDataset(torch.utils.data.Dataset):
    def __init__(self, df, AudioParams, duration=5, mode="train"):
        self.df = df
        self.AudioParams = AudioParams
        self.mode = mode
        self.duration = duration

        mean = (0.485, 0.456, 0.406)  # RGB
        std = (0.229, 0.224, 0.225)  # RGB

        self.albu_transforms = {
            "train": A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.OneOf(
                        [
                            A.Cutout(max_h_size=5, max_w_size=16),
                            A.CoarseDropout(max_holes=4),
                        ],
                        p=0.5,
                    ),
                    A.Normalize(mean, std),
                ]
            ),
            "valid": A.Compose(
                [
                    A.Normalize(mean, std),
                ]
            ),
        }

        if mode == "train":
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            NoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=5, max_snr=20),
                            PinkNoise(p=1, min_snr=5, max_snr=20),
                        ],
                        p=0.2,
                    ),
                    RandomVolume(p=0.2, limit=4),
                    Normalize(p=1),
                ]
            )
        else:
            self.wave_transforms = Compose(
                [
                    Normalize(p=1),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]

        wav_path = sample["file_path"]
        labels = sample["new_target"]

        y, _ = sf.read(wav_path, always_2d=True)
        if len(y.shape) > 1:  # there are (X, 2) arrays
            y = np.mean(y, 1)

        len_y = len(y)
        effective_length = self.AudioParams["sr"] * self.duration
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start : start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start : start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if len(y) > 0:
            y = y[: self.duration * self.AudioParams["sr"]]

            if self.wave_transforms:
                y = self.wave_transforms(y, sr=self.AudioParams["sr"])

        y = np.concatenate([y, y, y])[: self.duration * self.AudioParams["sr"]]
        y = crop_or_pad(
            y,
            self.duration * self.AudioParams["sr"],
            sr=self.AudioParams["sr"],
            train=True,
            probs=None,
        )
        image = compute_melspec(y, self.AudioParams)
        image = mono_to_color(image)
        image = image.astype(np.uint8)

        image = self.albu_transforms[self.mode](image=image)["image"]
        image = image.T

        targets = np.zeros(len(ALL_BIRDS), dtype=float)

        for ebird_code in labels.split():
            targets[ALL_BIRDS.index(ebird_code)] = 1.0

        return {
            "image": image,
            "targets": targets,
        }


class BirdCLEFDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, config):
        super().__init__()
        self.save_hyperparameters()
        self.train_df = train_df
        self.val_df = val_df
        self.config = config

    def setup(self, stage=None):

        # Create train dataset
        self.train_dataset = AllBirdsDataset(
            self.train_df,
            self.config["AudioParams"],
            mode="train",
        )

        # Create val dataset
        self.val_dataset = AllBirdsDataset(
            self.val_df,
            self.config["AudioParams"],
            mode="valid",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["train_bs"],
            num_workers=self.config["workers"],
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["valid_bs"],
            num_workers=self.config["workers"],
            shuffle=False,
            pin_memory=False,
        )
