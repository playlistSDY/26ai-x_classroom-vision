"""Inference-only BPBreID features for matching mutually visible body parts.

Adapted from VlSomers/bpbreid at a2dc4304284784d8b2c061764512f0698aa38c21.
Licensed under the Hippocratic License 3.0 (LAW-MEDIA-MIL-SOC-SV).
See vendor/NOTICE.md and vendor/LICENSE.bpbreid for attribution and license.
The verified Occluded-Duke checkpoint has eight parts, not five.
"""

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from vendor.hrnet import HighResolutionNet, get_hrnet_config


PART_NAMES = (
    "head", "left_arm", "right_arm", "torso",
    "left_leg", "right_leg", "left_foot", "right_foot",
)
PART_GROUPS = ("head", "arms", "arms", "torso", "legs", "legs", "feet", "feet")


def load_partial_checkpoint(path: str | Path):
    """Read tensors/config without executing the checkpoint's YACS classes.

    Historical checkpoints contain YACS config objects and NumPy scalar metrics.
    Remapping YACS to an inert OrderedDict keeps weights_only=True and avoids
    importing the upstream training stack or accepting arbitrary pickle globals.
    """
    numpy_core = np.__dict__.get("_core") or np.core
    allowed = [
        (OrderedDict, "yacs.config.CfgNode"),
        (numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype, type(np.dtype(np.float32)), type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


class _PixelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1920)
        self.classifier = nn.Conv2d(1920, len(PART_NAMES) + 1, 1)

    def forward(self, features):
        return self.classifier(self.bn(features))


class _PartDimReduction(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1920, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))

    def forward(self, features):
        batch, parts, _ = features.shape
        return self.layers(features.flatten(0, 1)).view(batch, parts, 512)


class PartialBodyModel(nn.Module):
    """Released HRNet-W32 + semantic attention + weighted average part pooling."""

    def __init__(self):
        super().__init__()
        self.backbone_appearance_feature_extractor = HighResolutionNet(get_hrnet_config(), False, 512)
        self.pixel_classifier = _PixelClassifier()
        self.parts_after_pooling_dim_reduce = _PartDimReduction()

    def forward(self, images, exclusion_masks=None, min_part_area=0.0025, min_part_confidence=0.35):
        spatial = self.backbone_appearance_feature_extractor(images)
        probabilities = self.pixel_classifier(spatial).float().softmax(dim=1)
        masks = probabilities[:, 1:]
        eligible = torch.ones_like(probabilities[:, 0], dtype=torch.bool)
        if exclusion_masks is not None:
            excluded = F.interpolate(exclusion_masks[:, None].float(), spatial.shape[-2:], mode="nearest")[:, 0] > 0.5
            eligible &= ~excluded
            masks = masks * eligible[:, None]

        # This is the published weighted average pooling, using bmm to avoid
        # materializing a large [N, K, D, H, W] intermediate on a 6 GB GPU.
        pooled = torch.bmm(masks.flatten(2), spatial.float().flatten(2).transpose(1, 2))
        pooled /= masks.sum(dim=(-2, -1)).clamp_min(1e-6)[:, :, None]
        parts = self.parts_after_pooling_dim_reduce(pooled.to(images.dtype)).float()
        parts = F.normalize(parts, dim=-1)

        # The original model marks a part visible if even one pixel predicts it.
        # Require meaningful confident support for live occlusion matching.
        predictions = probabilities.argmax(dim=1)
        support = torch.stack([(predictions == index + 1) & eligible
                               for index in range(len(PART_NAMES))], dim=1)
        counts = support.sum(dim=(-2, -1))
        area = counts / (spatial.shape[-2] * spatial.shape[-1])
        confidence = (masks * support).sum(dim=(-2, -1)) / counts.clamp_min(1)
        visible = (area >= min_part_area) & (confidence >= min_part_confidence)
        return parts, visible.float()


class PartialBodyEncoder:
    def __init__(self, device: torch.device, batch_size: int, checkpoint_path: str | Path,
                 min_part_area=0.0025, min_part_confidence=0.35):
        if batch_size < 1:
            raise ValueError("Partial ReID batch size must be positive.")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.min_part_area = min_part_area
        self.min_part_confidence = min_part_confidence
        weights = load_partial_checkpoint(checkpoint_path)
        config = weights["config"]["model"]["bpbreid"]
        expected = {"backbone": "hrnet32", "dim_reduce": "after_pooling",
                    "dim_reduce_output": 512, "pooling": "gwap", "normalization": "identity",
                    "learnable_attention_enabled": True, "test_use_target_segmentation": "none"}
        if any(config.get(key) != value for key, value in expected.items()) or config["masks"]["parts_num"] != len(PART_NAMES):
            raise ValueError("The BPBreID checkpoint architecture does not match this inference model.")
        self.model = PartialBodyModel()
        required = set(self.model.state_dict())
        state = {key.removeprefix("module."): value for key, value in weights["state_dict"].items()}
        inference_state = {key: value for key, value in state.items() if key in required}
        # Every inference parameter must be present, including BatchNorm buffers.
        # Only the original training classifiers and unused global heads are omitted.
        omitted = set(state) - required
        allowed_omissions = (
            "global_after_pooling_dim_reduce.", "foreground_after_pooling_dim_reduce.",
            "background_after_pooling_dim_reduce.", "global_identity_classifier.",
            "background_identity_classifier.", "foreground_identity_classifier.",
            "concat_parts_identity_classifier.", "parts_identity_classifier.",
        )
        if any(not key.startswith(allowed_omissions) for key in omitted):
            raise ValueError("Unexpected parameters in the partial ReID checkpoint.")
        self.model.load_state_dict(inference_state, strict=True)
        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(self, crops: list[np.ndarray], exclusion_masks: list[np.ndarray] | None = None):
        """Return (unit parts[8,512], visibility[8]) per BGR person crop.

        Optional masks share each crop's pixel coordinates; True excludes pixels
        occupied by other people. They restrict pooling and visibility support.
        """
        if exclusion_masks is not None and len(exclusion_masks) != len(crops):
            raise ValueError("Each partial ReID crop needs a matching exclusion mask.")
        output = []
        for start in range(0, len(crops), self.batch_size):
            selected = crops[start:start + self.batch_size]
            pixels = np.stack([cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (128, 384),
                                          interpolation=cv2.INTER_LINEAR) for crop in selected])
            images = torch.from_numpy(pixels).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            images = ((images - self.mean) / self.std).to(self.dtype)
            excluded = None
            if exclusion_masks is not None:
                selected_masks = exclusion_masks[start:start + self.batch_size]
                if any(mask.shape != crop.shape[:2] for mask, crop in zip(selected_masks, selected)):
                    raise ValueError("Exclusion masks must have the same height and width as their crops.")
                masks = np.stack([cv2.resize(mask.astype(np.uint8), (128, 384), interpolation=cv2.INTER_NEAREST)
                                  for mask in selected_masks])
                excluded = torch.from_numpy(masks).to(self.device)
            parts, visibility = self.model(images, excluded, self.min_part_area, self.min_part_confidence)
            output.extend(zip(parts.cpu().numpy(), visibility.cpu().numpy()))
        return output
