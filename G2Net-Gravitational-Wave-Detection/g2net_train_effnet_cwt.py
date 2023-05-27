#!/usr/bin/env python3

import torch
import gc
import timm
import wandb
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from nnAudio.Spectrogram import CQT1992v2
from scipy import signal, optimize
from timm.models.layers.conv2d_same import conv2d_same
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "/home/egortrushin/datasets/g2net-gravitational-wave-detection"
OUTPUT_DIR = "./Checkpoints/"

config = {
    "seed": 1,
    "fold_seed": 42,
    "train_folds": [0,1,2,3,4],
    "base_model": "tf_efficientnet_b1_ns",
    "base_model_classifier": "classifier",
    "classes": 1,
    "precision": 16,
    "train_batch_size": 64,
    "val_batch_size": 64,
    "epochs": 30,
    "num_workers": 4,
    "weight_decay": 1e-6,
    "lr": 3e-3,
    "min_lr": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "t_max": 30,
    "hop_length": [16],
    "bins_per_octave": [8],
    "fmin": 20,
    "fmax": 512,
    "image_size": (128, 256),
    "freq_sup": False,
    "logger": "wandb",
    "gradient_clip_val": 1.0e3,
}


def get_path(x, basedir):
    return f"{basedir}/{x[0]}/{x[1]}/{x[2]}/{x}.npy"


def apply_bandpass(x, lf=25, hf=500, order=4, sr=2048):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    return signal.sosfiltfilt(sos, x) / normalization

def get_logger(fold: int):
    return pl_loggers.WandbLogger(project='G2Net',
                                  config=config,
                                  name=f'b1_ns 3ch tests f{fold}',
                                  save_dir="./logs/")

# From https://github.com/tomrunia/PyTorchWavelets/blob/master/wavelets_pytorch/wavelets.py
class Morlet(object):
    def __init__(self, w0=6):
        """w0 is the nondimensional frequency constant. If this is
        set too low then the wavelet does not sample very well: a
        value over 5 should be ok; Terrence and Compo set it to 6.
        """
        self.w0 = w0
        if w0 == 6:
            # value of C_d from TC98
            self.C_d = 0.776

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0, complete=True):
        """
        Complex Morlet wavelet, centred at zero.
        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.
        complete : bool
            Whether to use the complete or the standard version.
        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given time
        See Also
        --------
        scipy.signal.gausspulse
        Notes
        -----
        The standard version::
            pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
        This commonly used wavelet is often referred to simply as the
        Morlet wavelet.  Note that this simplified version can cause
        admissibility problems at low values of `w`.
        The complete version::
            pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))
        The complete version of the Morlet wavelet, with a correction
        term to improve admissibility. For `w` greater than 5, the
        correction term is negligible.
        Note that the energy of the return wavelet is not normalised
        according to `s`.
        The fundamental frequency of this wavelet in Hz is given
        by ``f = 2*s*w*r / M`` where r is the sampling rate.
        """
        w = self.w0

        x = t / s

        output = np.exp(1j * w * x)

        if complete:
            output -= np.exp(-0.5 * (w ** 2))

        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output

    # Fourier wavelengths
    def fourier_period(self, s):
        """Equivalent Fourier period of Morlet"""
        return 4 * np.pi * s / (self.w0 + (2 + self.w0 ** 2) ** 0.5)

    def scale_from_period(self, period):
        """
        Compute the scale from the fourier period.
        Returns the scale
        """
        # Solve 4 * np.pi * scale / (w0 + (2 + w0 ** 2) ** .5)
        #  for s to obtain this formula
        coeff = np.sqrt(self.w0 * self.w0 + 2)
        return (period * (coeff + self.w0)) / (4.0 * np.pi)

    # Frequency representation
    def frequency(self, w, s=1.0):
        """Frequency representation of Morlet.
        Parameters
        ----------
        w : float
            Angular frequency. If `s` is not specified, i.e. set to 1,
            this can be used as the non-dimensional angular
            frequency w * s.
        s : float
            Scaling factor. Default is 1.
        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given frequency
        """
        x = w * s
        # Heaviside mock
        Hw = np.array(w)
        Hw[w <= 0] = 0
        Hw[w > 0] = 1
        return np.pi ** -0.25 * Hw * np.exp((-((x - self.w0) ** 2)) / 2)

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.
        This can be worked out analytically by solving
            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return 2 ** 0.5 * s


class CWT(nn.Module):
    def __init__(
        self,
        dj=0.0625,
        dt=1 / 2048,
        wavelet=Morlet(),
        fmin: int = 20,
        fmax: int = 500,
        output_format="Magnitude",
        trainable=False,
        hop_length: int = 1,
    ):
        super().__init__()
        self.wavelet = wavelet

        self.dt = dt
        self.dj = dj
        self.fmin = fmin
        self.fmax = fmax
        self.output_format = output_format
        self.trainable = trainable  # TODO make kernel a trainable parameter
        self.stride = (1, hop_length)
        # self.padding = 0  # "same"

        self._scale_minimum = self.compute_minimum_scale()

        self.signal_length = None
        self._channels = None

        self._scales = None
        self._kernel = None
        self._kernel_real = None
        self._kernel_imag = None

    def compute_optimal_scales(self):
        """
        Determines the optimal scale distribution (see. Torrence & Combo, Eq. 9-10).
        :return: np.ndarray, collection of scales
        """
        if self.signal_length is None:
            raise ValueError(
                "Please specify signal_length before computing optimal scales."
            )
        J = int(
            (1 / self.dj) * np.log2(self.signal_length * self.dt / self._scale_minimum)
        )
        scales = self._scale_minimum * 2 ** (self.dj * np.arange(0, J + 1))

        # Remove high and low frequencies
        frequencies = np.array([1 / self.wavelet.fourier_period(s) for s in scales])
        if self.fmin:
            frequencies = frequencies[frequencies >= self.fmin]
            scales = scales[0 : len(frequencies)]
        if self.fmax:
            frequencies = frequencies[frequencies <= self.fmax]
            scales = scales[len(scales) - len(frequencies) : len(scales)]

        return scales

    def compute_minimum_scale(self):
        """
        Choose s0 so that the equivalent Fourier period is 2 * dt.
        See Torrence & Combo Sections 3f and 3h.
        :return: float, minimum scale level
        """
        dt = self.dt

        def func_to_solve(s):
            return self.wavelet.fourier_period(s) - 2 * dt

        return optimize.fsolve(func_to_solve, 1)[0]

    def _build_filters(self):
        self._filters = []
        for scale_idx, scale in enumerate(self._scales):
            # Number of points needed to capture wavelet
            M = 10 * scale / self.dt
            # Times to use, centred at zero
            t = torch.arange((-M + 1) / 2.0, (M + 1) / 2.0) * self.dt
            if len(t) % 2 == 0:
                t = t[0:-1]  # requires odd filter size
            # Sample wavelet and normalise
            norm = (self.dt / scale) ** 0.5
            filter_ = norm * self.wavelet(t, scale)
            self._filters.append(torch.conj(torch.flip(filter_, [-1])))

        self._pad_filters()

    def _pad_filters(self):
        filter_len = self._filters[-1].shape[0]
        padded_filters = []

        for f in self._filters:
            pad = (filter_len - f.shape[0]) // 2
            padded_filters.append(nn.functional.pad(f, (pad, pad)))

        self._filters = padded_filters

    def _build_wavelet_bank(self):
        """This function builds a 2D wavelet filter using wavelets at different scales

        Returns:
            tensor: Tensor of shape (num_widths, 1, channels, filter_len)
        """
        self._build_filters()
        wavelet_bank = torch.stack(self._filters)
        wavelet_bank = wavelet_bank.view(
            wavelet_bank.shape[0], 1, 1, wavelet_bank.shape[1]
        )
        # See comment by tez6c32
        # https://www.kaggle.com/anjum48/continuous-wavelet-transform-cwt-in-pytorch/comments#1499878
        # wavelet_bank = torch.cat([wavelet_bank] * self.channels, 2)
        return wavelet_bank

    def forward(self, x):
        """Compute CWT arrays from a batch of multi-channel inputs

        Args:
            x (torch.tensor): Tensor of shape (batch_size, channels, time)

        Returns:
            torch.tensor: Tensor of shape (batch_size, channels, widths, time)
        """
        if self.signal_length is None:
            self.signal_length = x.shape[-1]
            self.channels = x.shape[-2]
            self._scales = self.compute_optimal_scales()
            self._kernel = self._build_wavelet_bank()

            if self._kernel.is_complex():
                self._kernel_real = self._kernel.real
                self._kernel_imag = self._kernel.imag

        x = x.unsqueeze(1)

        if self._kernel.is_complex():
            if (
                x.dtype != self._kernel_real.dtype
                or x.device != self._kernel_real.device
            ):
                self._kernel_real = self._kernel_real.to(device=x.device, dtype=x.dtype)
                self._kernel_imag = self._kernel_imag.to(device=x.device, dtype=x.dtype)

            # Strides > 1 not yet supported for "same" padding
            # output_real = nn.functional.conv2d(
            #     x, self._kernel_real, padding=self.padding, stride=self.stride
            # )
            # output_imag = nn.functional.conv2d(
            #     x, self._kernel_imag, padding=self.padding, stride=self.stride
            # )
            output_real = conv2d_same(x, self._kernel_real, stride=self.stride)
            output_imag = conv2d_same(x, self._kernel_imag, stride=self.stride)
            output_real = torch.transpose(output_real, 1, 2)
            output_imag = torch.transpose(output_imag, 1, 2)

            if self.output_format == "Magnitude":
                return torch.sqrt(output_real ** 2 + output_imag ** 2)
            else:
                return torch.stack([output_real, output_imag], -1)

        else:
            if x.device != self._kernel.device:
                self._kernel = self._kernel.to(device=x.device, dtype=x.dtype)

            # output = nn.functional.conv2d(
            #     x, self._kernel, padding=self.padding, stride=self.stride
            # )
            output = conv2d_same(x, self._kernel, stride=self.stride)
            return torch.transpose(output, 1, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.enable()

pl.seed_everything(config["seed"])

df = pd.read_csv(Path(DATA_DIR, "training_labels.csv"))

df["path"] = df.id.apply(get_path, basedir=Path(DATA_DIR, "train"))

Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["fold_seed"])
for n, (train_index, val_index) in enumerate(Fold.split(df, df["target"])):
    df.loc[val_index, "fold"] = int(n)
df["fold"] = df["fold"].astype(int)


class G2NetDataset(Dataset):
    def __init__(self, paths, targets=None):
        self.paths = paths
        self.targets = targets

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waves = np.load(self.paths[idx])
        for i in range(3):
            waves[i] = waves[i]*signal.tukey(4096, 0.2)
            waves[i] = apply_bandpass(waves[i], 35, 500) * 5.0e20
        waves = torch.from_numpy(waves).float()
        if self.targets is not None:
            return waves, torch.tensor(self.targets[idx], dtype=torch.long)
        else:
            return waves


class G2NetDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, config):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.config = config

    def setup(self, stage=None):

        # Create train dataset
        self.train_dataset = G2NetDataset(
            self.train_df.path.values, self.train_df.target.values
        )

        # Create val dataset
        self.val_dataset = G2NetDataset(
            self.val_df.path.values, self.val_df.target.values
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["train_batch_size"],
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


class G2NetClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.spec_layer1 = CWT(fmin=config["fmin"], fmax=config["fmax"], hop_length=config["hop_length"][0], dj=0.073/8)

        self.classifier = timm.create_model(self.hparams.base_model, pretrained=True, in_chans=3)
        n_features = self.classifier._modules[
            self.hparams.base_model_classifier
        ].in_features
        self.classifier._modules[self.hparams.base_model_classifier] = nn.Linear(
            n_features, self.hparams.classes
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):

        x = self.spec_layer1(x)
        #print(x.shape)
        x = nn.functional.interpolate(x, size=config["image_size"], mode='bicubic')

        #z0 = self.spec_layer2(x[:, 0, :])
        #z1 = self.spec_layer2(x[:, 1, :])
        #z2 = self.spec_layer2(x[:, 2, :])
        #x2 = torch.stack((z0, z1, z2), 1)
        #x2 = nn.functional.interpolate(x2, size=config["image_size"], mode='bicubic')

        #z0 = self.spec_layer3(x[:, 0, :])
        #z1 = self.spec_layer3(x[:, 1, :])
        #z2 = self.spec_layer3(x[:, 2, :])
        #x3 = torch.stack((z0, z1, z2), 1)
        #x3 = nn.functional.interpolate(x3, size=config["image_size"], mode='bicubic')

        #x = torch.cat((x1, x2, x3), 1)

        if config["freq_sup"]:
            freq_sup = torch.logspace(np.log10(config["fmin"]/config["fmax"]), 1.0, x.shape[2]).to(device)
            freq_sup = freq_sup[None,None,:,None].repeat(x.shape[0], x.shape[1], 1, 1)
            x = torch.cat((x, freq_sup), 3)
        out = self.classifier(x)
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(1)
        loss = self.loss(logits, labels.float())
        y_true = labels.cpu().numpy()
        y_pred = logits.cpu().detach().numpy()
        score = roc_auc_score(y_true, y_pred)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_score", score, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "train_score": score}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(1)
        loss = self.loss(logits, labels.float())
        y_true = labels.cpu().numpy()
        y_pred = logits.cpu().detach().numpy()
        score = roc_auc_score(y_true, y_pred)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_score", score, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "val_score": score}

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=self.hparams.min_lr,
            last_epoch=-1,
        )
        return {"optimizer": optimizer, "scheduler": scheduler}



for fold in config["train_folds"]:
    print(f"*** fold {fold} ***")

    train_df = df.loc[df.fold != fold]
    val_df = df.loc[df.fold == fold]

    data_module = G2NetDataModule(train_df, val_df, config)

    filename = f"{config['base_model']}-f{fold}-{{val_score:.5f}}-{{val_loss:.4f}}"

    checkpoint_callback = ModelCheckpoint(
        monitor="val_score", dirpath=OUTPUT_DIR, mode="max", filename=filename, save_top_k=2, verbose=1
    )
    early_stop_callback = EarlyStopping(monitor="val_score", mode="max", patience=8)
    
    if config["logger"] is not None:
        wandb_logger = get_logger(fold)
    else:
        wandb_logger = None

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config["epochs"],
        precision=config["precision"],
        deterministic=True,
        stochastic_weight_avg=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        #val_check_interval = 0.2,
        #limit_train_batches=0.1, limit_val_batches=0.1,
        logger = None#wandb_logger,
    )

    model = G2NetClassifier(config)
    trainer.fit(model, data_module)
    
    if config["logger"] is not None:
        wandb.finish()

    del model
    gc.collect()
