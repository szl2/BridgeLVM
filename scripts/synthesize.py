"""Generate deterioration-state images (T3.1) for the CorrosionCondition test images.

Example:
  python scripts/synthesize.py --model models/t3.1 --out outputs/t3.1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridgelvm.configs import SYNTHESIS as C  # noqa: E402
from bridgelvm.synthesis import NEGATIVE_PROMPT, CorrosionSynthesisDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="folder with the ControlNet LoRA adapter (e.g. models/t3.1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--base_model", default=C["base_model"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    from peft import PeftModel

    dev, dtype = "cuda", torch.float16
    controlnet = ControlNetModel.from_pretrained(C["controlnet"], torch_dtype=dtype)
    controlnet = PeftModel.from_pretrained(controlnet, a.model).merge_and_unload()
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        a.base_model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None).to(dev)
    pipe.set_progress_bar_config(disable=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = CorrosionSynthesisDataset(a.data_root, "Test", C["resolution"])
    for i in range(len(ds)):
        s = ds[i]
        src = Image.fromarray(((s["pixel_values"].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        cond = Image.fromarray((s["cond_map"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        with torch.inference_mode():
            img = pipe(prompt=s["prompt"], negative_prompt=NEGATIVE_PROMPT, image=src, control_image=cond,
                       strength=C["strength"], guidance_scale=C["guidance_scale"],
                       num_inference_steps=C["num_inference_steps"],
                       generator=torch.Generator(dev).manual_seed(a.seed)).images[0]
        img.save(out / f"{s['stem']}.png")
        print(f"[{i + 1}/{len(ds)}] {s['stem']}", flush=True)
    print(f"{len(ds)} images saved to {out}")


if __name__ == "__main__":
    main()
