"""HF Trainer setup for the segmentation tasks (T2.1, T2.2): class-weighted cross-entropy + Dice loss."""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, Trainer, TrainingArguments

from .configs import SEGFORMER

_processor = None


def collate_fn(examples):
    """Resize to 512x512 and normalise images; binarise masks (>0 -> 1)."""
    global _processor
    if _processor is None:
        _processor = SegformerImageProcessor.from_pretrained(SEGFORMER)
    enc = _processor(images=[e["image"] for e in examples],
                     segmentation_maps=[(np.array(e["mask"]) > 0).astype(np.int64) for e in examples],
                     return_tensors="pt")
    enc["labels"] = enc["labels"].long()
    return dict(enc)


def _predictions(logits, labels):
    up = F.interpolate(torch.from_numpy(logits).float(), size=labels.shape[-2:], mode="bilinear",
                       align_corners=False)
    return up.argmax(dim=1).numpy()


def _iou(pred, true):
    union = np.logical_or(pred, true).sum()
    return float(np.logical_and(pred, true).sum() / (union + 1e-6)) if union > 0 else 1.0


def crack_metrics(eval_pred):
    logits, labels = eval_pred
    preds = _predictions(logits, labels)
    iou = [_iou(preds == c, labels == c) for c in (0, 1)]
    return {"iou_background": iou[0], "iou_crack": iou[1], "mean_iou": float(np.mean(iou))}


def corrosion_metrics(eval_pred):
    logits, labels = eval_pred
    preds = _predictions(logits, labels)
    m = {}
    for c, name in enumerate(["background", "corrosion"]):
        p, t = preds == c, labels == c
        m[f"iou_{name}"] = _iou(p, t)
        tp = np.logical_and(p, t).sum()
        fp = np.logical_and(p, ~t).sum()
        fn = np.logical_and(~p, t).sum()
        m[f"f1_{name}"] = float(2 * tp / (2 * tp + fp + fn + 1e-6))
    m["mean_iou"] = float(np.mean([m["iou_background"], m["iou_corrosion"]]))
    m["macro_f1"] = float(np.mean([m["f1_background"], m["f1_corrosion"]]))
    return m


def dice_loss(logits, labels, num_classes=2, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    card = (probs + onehot).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (card + smooth)).mean()


class CeDiceTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def _wrap_model(self, model, training=True, dataloader=None):
        return model        # no DataParallel (incompatible with PEFT modules_to_save)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        up = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        ce = F.cross_entropy(up, labels, weight=self.class_weights.to(up.device, dtype=up.dtype))
        loss = ce + dice_loss(up, labels)
        return (loss, outputs) if return_outputs else loss


def add_deep_supervision(trainer):
    """DeepCrack: each of the five side outputs adds a class-weighted cross-entropy loss (weight 1.0)."""
    base = trainer.compute_loss

    def compute_loss(model, inputs, return_outputs=False, **kw):
        labels = inputs["labels"]
        loss, outputs = base(model, dict(inputs), return_outputs=True, **kw)
        net = model.module if hasattr(model, "module") else model
        sides = getattr(net, "_side_logits", None)
        if model.training and sides:
            w = trainer.class_weights.to(sides[0].device, dtype=sides[0].dtype)
            for s in sides:
                lab = labels if s.shape[-2:] == labels.shape[-2:] else F.interpolate(
                    labels[:, None].float(), size=s.shape[-2:], mode="nearest")[:, 0].long()
                loss = loss + F.cross_entropy(s, lab, weight=w)
            net._side_logits = None
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = compute_loss
    return trainer


def build_trainer(cfg, model, train_ds, val_ds, output_dir, seed, class_weights):
    args = TrainingArguments(
        output_dir=output_dir, learning_rate=cfg["lr"], per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=cfg["batch"], num_train_epochs=cfg["epochs"], weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup"], lr_scheduler_type=cfg["scheduler"], eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model=cfg["select_metric"], greater_is_better=True,
        fp16=torch.cuda.is_available(), logging_steps=20, remove_unused_columns=False, label_names=["labels"],
        seed=seed, report_to="none", save_total_limit=3, dataloader_num_workers=0, dataloader_pin_memory=False)
    trainer = CeDiceTrainer(class_weights=class_weights, model=model, args=args, train_dataset=train_ds,
                            eval_dataset=val_ds, data_collator=collate_fn,
                            compute_metrics=crack_metrics if cfg["metrics"] == "crack" else corrosion_metrics)
    return add_deep_supervision(trainer) if cfg["arm"] == "deepcrack" else trainer
