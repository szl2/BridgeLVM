"""Datasets for the classification and segmentation tasks.

Every task reads its fixed train / validation / test split from splits/<task>.json.
Paths in the split files are relative to the dataset folder (data/<Dataset>/).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .configs import DATASET_DIR

REPO = Path(__file__).resolve().parents[1]


def load_split(task, split):
    """List of entries of one split: [image, label] (classification) or [image, mask] (segmentation)."""
    return json.loads((REPO / "splits" / f"{task}.json").read_text())[split]


class ClassificationDataset(Dataset):
    def __init__(self, task, split, data_root, transform=None, limit=0):
        root = Path(data_root) / DATASET_DIR[task]
        entries = load_split(task, split)
        if limit:
            entries = entries[:limit]
        self.paths = [root / p for p, _ in entries]
        self.labels = [int(y) for _, y in entries]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"pixel_values": image, "labels": self.labels[i]}


class SegmentationDataset(Dataset):
    """Returns PIL image and binary PIL mask; the collate function resizes and normalises."""

    def __init__(self, task, split, data_root, limit=0):
        root = Path(data_root) / DATASET_DIR[task]
        entries = load_split(task, split)
        if limit:
            entries = entries[:limit]
        self.pairs = [(root / i, root / m) for i, m in entries]
        self.rgb_mask = task == "t2.2"     # CorrosionCondition masks are RGB colour codes

    def __len__(self):
        return len(self.pairs)

    def mask_array(self, i):
        path = self.pairs[i][1]
        if self.rgb_mask:   # any non-black pixel (fair, poor or severe corrosion) -> 1
            return (np.array(Image.open(path).convert("RGB")).sum(axis=2) > 0).astype(np.uint8)
        return np.array(Image.open(path).convert("L"))

    def __getitem__(self, i):
        image = Image.open(self.pairs[i][0]).convert("RGB")
        mask = Image.fromarray(self.mask_array(i), mode="L")
        return {"image": image, "mask": mask}


def classification_transforms(cfg):
    """(train, eval) torchvision transforms for a classification recipe."""
    from torchvision.transforms import (CenterCrop, ColorJitter, Compose, Normalize, RandomHorizontalFlip,
                                        RandomResizedCrop, RandomRotation, Resize, ToTensor)
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(cfg["model_id"])
    size = proc.size.get("height", proc.size.get("shortest_edge", 224))
    norm = Normalize(mean=proc.image_mean, std=proc.image_std)
    if cfg["arm"] in ("vgg16", "alexnet"):          # torchvision ImageNet weights expect ImageNet statistics
        size, norm = 224, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if cfg["transforms"] == "default":
        train = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), norm])
    else:   # "strong"
        train = Compose([RandomResizedCrop(size, scale=(0.7, 1.0)), RandomHorizontalFlip(), RandomRotation(15),
                         ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), ToTensor(), norm])
    evaluate = Compose([Resize(size + 32), CenterCrop(size), ToTensor(), norm])
    return train, evaluate


def classification_weights(labels, power):
    """Class weights ('balanced' weights raised to `power`, normalised to mean 1)."""
    import torch
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array(sorted(set(labels)))
    w = compute_class_weight("balanced", classes=classes, y=np.array(labels)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def segmentation_weights(ds):
    """Inverse pixel-frequency class weights (background, foreground), normalised to mean 1."""
    import torch
    fg = tot = 0
    for i in range(len(ds)):
        b = ds.mask_array(i) > 0
        fg += int(b.sum())
        tot += b.size
    f = fg / tot
    w_bg, w_fg = 1.0 / (1.0 - f), 1.0 / f
    m = (w_bg + w_fg) / 2
    return torch.tensor([w_bg / m, w_fg / m], dtype=torch.float32)
