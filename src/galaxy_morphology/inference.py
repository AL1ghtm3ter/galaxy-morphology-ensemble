"""Five-view inference and validated ensemble weights."""
import torch


def normalize_weights(weights, count):
    values = [1.0] * count if weights is None else list(weights)
    if len(values) != count or count == 0:
        raise ValueError("Provide one weight for each model.")
    if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in values):
        raise ValueError("Weights must be finite and nonnegative.")
    total = sum(values)
    if total <= 0:
        raise ValueError("At least one ensemble weight must be positive.")
    return [value / total for value in values]


def five_view_prediction(model, images):
    views = [
        images,
        images.flip([3]),
        images.flip([2]),
        images.rot90(1, [2, 3]),
        images.rot90(2, [2, 3]),
    ]
    return torch.stack([model(view) for view in views]).mean(dim=0)


@torch.inference_mode()
def predict_batch(models, images, weights=None, tta=True):
    weights = normalize_weights(weights, len(models))
    prediction = None
    for model, weight in zip(models, weights):
        model.eval()
        value = five_view_prediction(model, images) if tta else model(images)
        if not torch.isfinite(value).all():
            raise ValueError("Model produced nonfinite outputs.")
        contribution = value * weight
        prediction = contribution if prediction is None else prediction + contribution
    return prediction

