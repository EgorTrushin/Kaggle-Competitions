#!/usr/bin/env python3

import random
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ContrailsDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_size=256, train=True, flips=False):
        self.df = df
        self.trn = train
        self.normalize_image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_size = image_size
        if image_size != 256:
            self.resize_image = T.transforms.Resize(image_size)
        self.flips = flips

    def __getitem__(self, index):
        row = self.df.iloc[index]
        con_path = row.path
        con = np.load(str(con_path))

        img = con[..., :-1]
        label = con[..., -1]

        label = torch.tensor(label)

        img = torch.tensor(np.reshape(img, (256, 256, 3))).to(torch.float32).permute(2, 0, 1)

        if self.trn and self.flips:
            if random.random() < 0.5:
                img = TF.vflip(img)
                label = TF.vflip(label)
            if random.random() < 0.5:
                img = TF.hflip(img)
                label = TF.hflip(label)

        if self.image_size != 256:
            img = self.resize_image(img)

        img = self.normalize_image(img)

        return img.float(), label.float()

    def __len__(self):
        return len(self.df)
