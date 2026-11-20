"""T3.1 deterioration-state synthesis: CorrosionCondition images with their colour-coded severity masks.

Mask colours: (0,0,0) background, (128,0,0) fair, (0,128,0) poor, (128,128,0) severe corrosion.
The original Train/ (396 images) and Test/ (44 images) folders of the 512x512 release are used.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .configs import SYNTHESIS

PROMPT = ("photograph of a bridge steel component, {delta}, realistic rust and corrosion spread, "
          "preserve structural geometry, high detail, natural lighting")
NEGATIVE_PROMPT = "painting, illustration, cartoon, low quality, blurry, unrealistic, deformed, text, watermark"


class CorrosionSynthesisDataset(Dataset):
    def __init__(self, data_root, split="Train", resolution=512, delta_text=SYNTHESIS["delta_text"]):
        root = Path(data_root) / "CorrosionCondition" / split
        self.img_dir, self.msk_dir = root / "images_512", root / "mask_512"
        self.stems = sorted(p.stem for p in self.img_dir.iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and (self.msk_dir / f"{p.stem}.png").exists())
        self.prompt = PROMPT.format(delta=delta_text)
        self.img_tf = transforms.Compose([   # [-1, 1] for the VAE
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        self.cond_tf = transforms.Compose([  # [0, 1] for the ControlNet
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.stems)

    def _image(self, stem):
        for ext in (".jpeg", ".jpg", ".png"):
            p = self.img_dir / f"{stem}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(stem)

    def __getitem__(self, i):
        stem = self.stems[i]
        return {"pixel_values": self.img_tf(self._image(stem)),
                "cond_map": self.cond_tf(Image.open(self.msk_dir / f"{stem}.png").convert("RGB")),
                "prompt": self.prompt, "stem": stem}
