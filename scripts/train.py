"""Train BridgeLVM (LoRA) or a reference model on one task and report its test scores.

Examples:
  python scripts/train.py --task t2.1 --out runs/t2.1                      # BridgeLVM
  python scripts/train.py --task t1.3 --arm vgg16 --out runs/ref_t1.3     # reference model
  python scripts/train.py --task t1.1 --limit 8 --epochs 1 --out runs/smoke   # quick check
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridgelvm import configs  # noqa: E402
from bridgelvm.data import (ClassificationDataset, SegmentationDataset, classification_transforms,  # noqa: E402
                            classification_weights, segmentation_weights)
from bridgelvm.models import build_model  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scores(metrics):
    return {k: round(v, 6) for k, v in metrics.items()
            if k.startswith("test_") and isinstance(v, float) and k != "test_loss"
            and not any(s in k for s in ("runtime", "per_second", "preparation"))}


def train_hf(cfg, a, out):
    """BridgeLVM and the VGG-16 / DeepCrack / DeepLabV3+ references (HF Trainer)."""
    if cfg["family"] == "cls":
        from bridgelvm.train_cls import build_trainer
        train_tf, eval_tf = classification_transforms(cfg)
        tr = ClassificationDataset(a.task, "train", a.data_root, train_tf, a.limit)
        va = ClassificationDataset(a.task, "val", a.data_root, eval_tf, a.limit)
        te = ClassificationDataset(a.task, "test", a.data_root, eval_tf, a.limit)
        cw = classification_weights(tr.labels, cfg["weight_power"]) if cfg["trainer"] == "weighted" else None
    else:
        from bridgelvm.train_seg import build_trainer
        tr, va, te = (SegmentationDataset(a.task, s, a.data_root, a.limit) for s in ("train", "val", "test"))
        cw = segmentation_weights(tr)
    print(f"train / val / test: {len(tr)} / {len(va)} / {len(te)}")

    model = build_model(cfg)
    trainer = build_trainer(cfg, model, tr, va, str(out / "checkpoints"), a.seed, cw)
    trainer.train()
    test = trainer.evaluate(te, metric_key_prefix="test")

    if cfg["arm"] == "lora":
        model.save_pretrained(str(out / "model"))
        if cfg["family"] == "seg":
            bn = {k: v.detach().cpu() for k, v in model.state_dict().items()
                  if "batch_norm" in k and ("running_" in k or "num_batches" in k)}
            torch.save(bn, out / "model" / "bn_stats.pt")
    else:
        torch.save(model.state_dict(), out / "model.pt")
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    return scores(test)


def train_alexnet(cfg, a, out):
    """AlexNet reference for T1.1 (crack-preserving augmentation, plain PyTorch loop)."""
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop,
                                        RandomVerticalFlip, Resize, ToTensor)
    from bridgelvm.models import AlexNetClassifier

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = Compose([RandomResizedCrop(224, scale=(0.7, 1.0)), RandomHorizontalFlip(), RandomVerticalFlip(),
                        ToTensor(), norm])
    eval_tf = Compose([Resize((224, 224)), ToTensor(), norm])
    tr = ClassificationDataset(a.task, "train", a.data_root, train_tf, a.limit)
    va = ClassificationDataset(a.task, "val", a.data_root, eval_tf, a.limit)
    te = ClassificationDataset(a.task, "test", a.data_root, eval_tf, a.limit)
    workers = 8 if dev == "cuda" else 0
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=cfg["batch"], shuffle=shuffle, num_workers=workers,
                                            pin_memory=dev == "cuda")
    dtr, dva, dte = loader(tr, True), loader(va, False), loader(te, False)

    m = AlexNetClassifier(len(cfg["labels"])).net.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(dtr)
    warm = int(cfg["warmup"] * total)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    @torch.no_grad()
    def accuracy(dl, labels):
        m.eval()
        preds = []
        for b in dl:
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                preds.append(m(b["pixel_values"].to(dev)).argmax(1).cpu())
        return (torch.cat(preds) == torch.tensor(labels)).float().mean().item()

    best, best_state = -1.0, None
    for ep in range(cfg["epochs"]):
        m.train()
        for b in dtr:
            x, y = b["pixel_values"].to(dev), b["labels"].to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = F.cross_entropy(m(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
        acc = accuracy(dva, va.labels)
        print(f"epoch {ep + 1}: val accuracy {acc:.4f}", flush=True)
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    torch.save(m.state_dict(), out / "model.pt")
    return {"test_accuracy": round(accuracy(dte, te.labels), 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(configs.TASKS))
    ap.add_argument("--arm", default="lora", choices=["lora", *configs.REFERENCES],
                    help="lora = BridgeLVM; otherwise a reference model (vgg16: t1.3, alexnet: t1.1, "
                         "deepcrack: t2.1, deeplabv3p: t2.2)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="override the number of epochs (testing)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images of each split (testing)")
    a = ap.parse_args()

    cfg = configs.get(a.task, a.arm)
    if a.epochs:
        cfg["epochs"] = a.epochs
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    test = train_alexnet(cfg, a, out) if a.arm == "alexnet" else train_hf(cfg, a, out)
    (out / "results.json").write_text(json.dumps({"task": a.task, "arm": a.arm, "seed": a.seed, "config": cfg,
                                                  "test": test}, indent=1))
    print(json.dumps(test, indent=1))


if __name__ == "__main__":
    main()
