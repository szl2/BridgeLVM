"""Evaluate a trained BridgeLVM model on the test split of its task.

Example:
  python scripts/evaluate.py --task t2.1 --model models/t2.1
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridgelvm import configs  # noqa: E402
from bridgelvm.data import ClassificationDataset, SegmentationDataset, classification_transforms  # noqa: E402
from bridgelvm.models import load_trained  # noqa: E402

MAIN_METRIC = {"t1.1": "accuracy", "t1.2": "accuracy", "t1.3": "accuracy", "t2.1": "mean_iou", "t2.2": "macro_f1"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(configs.TASKS))
    ap.add_argument("--model", required=True, help="folder with the LoRA adapter (e.g. models/t2.1)")
    ap.add_argument("--data_root", default="data")
    a = ap.parse_args()

    cfg = configs.get(a.task)
    tmp = tempfile.mkdtemp()
    model = load_trained(cfg, a.model)
    if cfg["family"] == "cls":
        from bridgelvm.train_cls import build_trainer
        _, eval_tf = classification_transforms(cfg)
        test = ClassificationDataset(a.task, "test", a.data_root, eval_tf)
        trainer = build_trainer(cfg, model, None, test, tmp, 0, class_weights=torch.ones(len(cfg["labels"])))
    else:
        from bridgelvm.train_seg import build_trainer
        test = SegmentationDataset(a.task, "test", a.data_root)
        trainer = build_trainer(cfg, model, None, test, tmp, 0, class_weights=torch.ones(2))

    res = trainer.evaluate(test, metric_key_prefix="test")
    res = {k[5:]: round(v, 4) for k, v in res.items()
           if isinstance(v, float) and k != "test_loss" and not any(s in k for s in ("runtime", "per_second", "preparation"))}
    print(json.dumps(res, indent=1))
    print(f"{a.task} test {MAIN_METRIC[a.task]}: {100 * res[MAIN_METRIC[a.task]]:.2f}%  ({len(test)} images)")


if __name__ == "__main__":
    main()
