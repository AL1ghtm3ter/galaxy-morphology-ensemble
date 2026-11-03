"""Training and validation without silently dropping failed batches."""
from contextlib import nullcontext

import numpy as np
import torch

from .inference import predict_batch
from .metrics import regression_metrics


def autocast_context(device, enabled):
    return torch.autocast("cuda") if enabled and device.type == "cuda" else nullcontext()


def train_epoch(model, loader, optimizer, criterion, device, accumulation_steps=4,
                gradient_clip=1.0, amp=True):
    if accumulation_steps < 1 or len(loader) == 0:
        raise ValueError("Use a nonempty loader and a positive accumulation step count.")
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_samples, accumulated_samples = 0.0, 0, 0
    for index, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        count = len(images)
        with autocast_context(device, amp):
            loss = criterion(model(images), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss in batch {index}.")
        # Sum sample contributions, then normalize once per optimizer step.
        # This also handles a short last batch or partial accumulation group.
        scaler.scale(loss * count).backward()
        accumulated_samples += count
        total_samples += count
        total_loss += loss.item() * count
        if (index + 1) % accumulation_steps == 0 or index + 1 == len(loader):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0
    return total_loss / total_samples


@torch.inference_mode()
def collect_predictions(models, loader, device, weights=None, tta=True, amp=True):
    predictions, targets = [], []
    for images, labels in loader:
        with autocast_context(device, amp):
            values = predict_batch(models, images.to(device), weights, tta)
        predictions.append(values.float().cpu().numpy())
        targets.append(labels.numpy())
    if not predictions:
        raise ValueError("Cannot evaluate an empty dataset.")
    return np.concatenate(targets), np.concatenate(predictions)


def evaluate_models(models, loader, device, weights=None, tta=True, amp=True):
    targets, predictions = collect_predictions(models, loader, device, weights, tta, amp)
    return regression_metrics(targets, predictions)

