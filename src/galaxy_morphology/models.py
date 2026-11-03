"""Three torchvision backbones with the coursework's independent task heads."""
import torch
from torch import nn
from torchvision import models

from .tasks import TASK_SIZES

ARCHITECTURES = ("vit", "swin", "efficientnet")


class MultiTaskRegressor(nn.Module):
    def __init__(self, backbone, feature_size):
        super().__init__()
        self.backbone = backbone
        for number, size in enumerate(TASK_SIZES, start=1):
            # Retain the original task1 ... task11 state-dictionary names.
            setattr(self, f"task{number}", nn.Linear(feature_size, size))

    def forward(self, images):
        features = self.backbone(images)
        return torch.cat([
            getattr(self, f"task{number}")(features)
            for number in range(1, len(TASK_SIZES) + 1)
        ], dim=1)


def build_model(architecture, pretrained=True):
    if architecture == "vit":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.vit_b_16(weights=weights)
        features = backbone.heads.head.in_features
        backbone.heads.head = nn.Identity()
    elif architecture == "swin":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.swin_t(weights=weights)
        features = backbone.head.in_features
        backbone.head = nn.Identity()
    elif architecture == "efficientnet":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_v2_s(weights=weights)
        features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()
    else:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {ARCHITECTURES}.")
    return MultiTaskRegressor(backbone, features)

