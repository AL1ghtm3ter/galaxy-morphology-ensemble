"""CSV validation, deterministic splits and strict image loading."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .tasks import LABEL_COLUMNS


def read_labels(data_dir, limit=None):
    path = Path(data_dir) / "training_solutions_rev1.csv"
    frame = pd.read_csv(path)
    required = ["GalaxyID", *LABEL_COLUMNS]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing target columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, required].copy()
    if frame.empty or frame["GalaxyID"].isna().any() or frame["GalaxyID"].duplicated().any():
        raise ValueError("GalaxyID values must be present, unique and nonempty.")
    ids = pd.to_numeric(frame["GalaxyID"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("GalaxyID values must be finite integers.")
    frame["GalaxyID"] = ids.astype(np.int64)
    values = frame.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Vote fractions must be finite and lie between zero and one.")
    if limit is not None:
        if limit < 4:
            raise ValueError("A smoke run needs at least four samples.")
        frame = frame.head(limit)
    return frame.reset_index(drop=True)


def split_labels(frame, seed=42, validation_fraction=0.2):
    if not 0 < validation_fraction < 1 or len(frame) < 4:
        raise ValueError("Use at least four samples and a validation fraction between zero and one.")
    train, validation = train_test_split(frame, test_size=validation_fraction, random_state=seed)
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def image_transform(training=False):
    steps = []
    if training:
        steps.extend([
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
    steps.extend([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(steps)


class GalaxyDataset(Dataset):
    def __init__(self, frame, image_dir, training=False):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.image_dir / f"{int(row['GalaxyID'])}.jpg"
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Unable to load galaxy image {path.name}") from error
        targets = torch.from_numpy(row.loc[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32))
        return tensor, targets

