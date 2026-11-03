"""Explicit, self-describing checkpoints for the refactored command-line tools."""
from pathlib import Path

import torch

from .models import build_model
from .tasks import LABEL_COLUMNS


def save_checkpoint(path, model, architecture, validation_ids, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **metadata,
        "format_version": 1,
        "architecture": architecture,
        "model_state": model.state_dict(),
        "validation_ids": [int(value) for value in validation_ids],
        "label_columns": list(LABEL_COLUMNS),
        "image_size": 224,
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or "architecture" not in checkpoint:
        raise ValueError("Expected a CLI checkpoint. Raw coursework state dictionaries use the reference notebook.")
    if checkpoint.get("label_columns") != list(LABEL_COLUMNS):
        raise ValueError("Checkpoint label order differs from the Galaxy Zoo target order.")
    model = build_model(checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint

