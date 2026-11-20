"""Put the CODEBRIM images used for T1.3 into data/CODEBRIM/images/ as 640x640 JPEGs.

Works with either download (see data/README.md):
  official release (Zenodo):  the unzipped original-images folder, which contains images/ and annotations/
  Roboflow export:            the unzipped export folder (train/ valid/ test/)

Example:
  python scripts/prepare_codebrim.py --source ~/Downloads/CODEBRIM_original_images
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="unzipped CODEBRIM download (Zenodo or Roboflow)")
    ap.add_argument("--out", default=str(REPO / "data" / "CODEBRIM"))
    a = ap.parse_args()

    split = json.loads((REPO / "splits" / "t1.3.json").read_text())
    needed = {Path(p).stem for s in split.values() for p, _ in s}
    out = Path(a.out) / "images"
    out.mkdir(parents=True, exist_ok=True)

    found = {}
    for p in Path(a.source).rglob("*.jpg"):
        stem = re.sub(r"_jpg\.rf\..*$", "", p.stem)      # Roboflow names: image_0000005_jpg.rf.<hash>.jpg
        if stem in needed:
            found[stem] = p
    missing = needed - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} of {len(needed)} images not found under {a.source}, "
                         f"e.g. {sorted(missing)[:3]}")

    for stem, p in sorted(found.items()):
        dst = out / f"{stem}.jpg"
        if ".rf." in p.name:                               # Roboflow images are already 640x640
            shutil.copyfile(p, dst)
        else:                                              # original images: orient and resize to 640x640
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            im.resize((640, 640), Image.BILINEAR).save(dst, quality=95)
    print(f"{len(found)} images written to {out}")


if __name__ == "__main__":
    main()
