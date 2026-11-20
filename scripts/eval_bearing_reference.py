"""Evaluate the released BearingCondition reference model (EfficientNet-B3, https://doi.org/10.7294/16628698.v1)
on the 300x300 test images of the dataset release (data/BearingCondition/Test_300x300).

The model definition and preprocessing follow the authors' code: 300x300 resize, normalisation with the
per-channel mean/std of the training images.

Example:
  python scripts/eval_bearing_reference.py --checkpoint "<unzipped model>/l2_loss/checkpoint_9.pt"
"""

import argparse
from pathlib import Path

import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


class Classifier(nn.Module):
    """EfficientNet-B3 classifier of the released model."""

    def __init__(self):
        super().__init__()
        from efficientnet_pytorch import EfficientNet
        self.resnet = EfficientNet.from_pretrained("efficientnet-b3")
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(256, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet(x).view(x.size(0), -1)
        return self.l2(self.dropout(self.relu(self.l1(x))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint_9.pt of the released model")
    ap.add_argument("--data_root", default="data")
    a = ap.parse_args()

    root = Path(a.data_root) / "BearingCondition"
    plain = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
    train = torchvision.datasets.ImageFolder(str(root / "Train_300x300"), transform=plain)
    mean = std = 0.0
    for x, _ in DataLoader(train, batch_size=8):
        flat = x.view(x.size(0), x.size(1), -1)
        mean = mean + flat.mean(2).sum(0)
        std = std + flat.std(2).sum(0)
    mean, std = (mean / len(train)).numpy(), (std / len(train)).numpy()

    tf = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test = torchvision.datasets.ImageFolder(str(root / "Test_300x300"), transform=tf)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Classifier().to(dev)
    model.load_state_dict(torch.load(a.checkpoint, map_location=dev, weights_only=False))
    model.eval()
    y, p = [], []
    with torch.no_grad():
        for x, t in DataLoader(test, batch_size=8):
            p += model(x.to(dev)).argmax(1).cpu().tolist()
            y += t.tolist()
    print(f"test images: {len(y)}")
    print(f"accuracy: {100 * accuracy_score(y, p):.2f}%   macro F1: {100 * f1_score(y, p, average='macro'):.2f}%")


if __name__ == "__main__":
    main()
