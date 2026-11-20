"""HF Trainer setup for the classification tasks (T1.1-T1.3)."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments


def collate_fn(examples):
    return {"pixel_values": torch.stack([e["pixel_values"] for e in examples]),
            "labels": torch.tensor([e["labels"] for e in examples], dtype=torch.long)}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, pred), "f1_macro": f1_score(labels, pred, average="macro")}


class WeightedTrainer(Trainer):
    """Class-weighted cross-entropy with label smoothing."""

    def __init__(self, class_weights, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device, dtype=logits.dtype),
                                   label_smoothing=self.label_smoothing)(logits, inputs["labels"])
        return (loss, outputs) if return_outputs else loss


def build_trainer(cfg, model, train_ds, val_ds, output_dir, seed, class_weights=None):
    args = dict(output_dir=output_dir, learning_rate=cfg["lr"], per_device_train_batch_size=cfg["batch"],
                per_device_eval_batch_size=cfg["batch"], num_train_epochs=cfg["epochs"],
                weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup"], eval_strategy="epoch",
                save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model=cfg["select_metric"],
                greater_is_better=True, fp16=torch.cuda.is_available(), remove_unused_columns=False,
                label_names=["labels"], seed=seed, report_to="none")
    common = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate_fn,
                  compute_metrics=compute_metrics)
    if cfg["trainer"] == "plain":
        return Trainer(args=TrainingArguments(logging_steps=50, save_total_limit=2, **args), **common)
    args = TrainingArguments(lr_scheduler_type=cfg["scheduler"], lr_scheduler_kwargs={"num_cycles": cfg["num_cycles"]},
                             logging_steps=20, save_total_limit=3, **args)
    return WeightedTrainer(class_weights=class_weights, label_smoothing=cfg["label_smoothing"], args=args, **common)
