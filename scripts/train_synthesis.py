"""Train the T3.1 ControlNet LoRA (Stable Diffusion v1.5 + segmentation ControlNet) on CorrosionCondition.

Example:
  python scripts/train_synthesis.py --out runs/t3.1
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridgelvm.configs import SYNTHESIS as C  # noqa: E402
from bridgelvm.synthesis import CorrosionSynthesisDataset  # noqa: E402


def collate(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "cond_map": torch.stack([b["cond_map"] for b in batch]),
            "prompt": [b["prompt"] for b in batch]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (the LoRA adapter is saved in <out>/model)")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--base_model", default=C["base_model"])
    ap.add_argument("--steps", type=int, default=C["steps"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.manual_seed(a.seed)
    dev, dtype = torch.device("cuda"), torch.float16
    vae = AutoencoderKL.from_pretrained(a.base_model, subfolder="vae", torch_dtype=dtype).to(dev).requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained(a.base_model, subfolder="tokenizer")
    text = CLIPTextModel.from_pretrained(a.base_model, subfolder="text_encoder", torch_dtype=dtype).to(dev).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(a.base_model, subfolder="unet", torch_dtype=dtype).to(dev).requires_grad_(False)
    sched = DDPMScheduler.from_pretrained(a.base_model, subfolder="scheduler")
    controlnet = ControlNetModel.from_pretrained(C["controlnet"], torch_dtype=dtype)
    controlnet = get_peft_model(controlnet, LoraConfig(r=C["lora_r"], lora_alpha=C["lora_alpha"],
                                                       lora_dropout=C["lora_dropout"], bias="none",
                                                       target_modules=C["target_modules"])).to(dev)
    controlnet.print_trainable_parameters()
    vae.eval(); text.eval(); unet.eval(); controlnet.train()

    ds = CorrosionSynthesisDataset(a.data_root, "Train", C["resolution"])
    loader = DataLoader(ds, batch_size=C["batch"], shuffle=True, collate_fn=collate, drop_last=True)
    params = [p for p in controlnet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=C["lr"])
    scaler = torch.amp.GradScaler("cuda")

    step = 0
    while step < a.steps:
        for b in loader:
            if step >= a.steps:
                break
            with torch.no_grad():
                lat = vae.encode(b["pixel_values"].to(dev, dtype)).latent_dist.sample() * 0.18215
                ids = tok(b["prompt"], padding="max_length", truncation=True, max_length=tok.model_max_length,
                          return_tensors="pt").input_ids.to(dev)
                enc = text(ids)[0]
            noise = torch.randn_like(lat)
            t = torch.randint(0, sched.config.num_train_timesteps, (lat.size(0),), device=dev).long()
            noisy = sched.add_noise(lat, noise, t)
            with torch.amp.autocast("cuda", dtype=dtype):
                down, mid = controlnet(noisy, t, encoder_hidden_states=enc,
                                       controlnet_cond=b["cond_map"].to(dev, torch.float32), return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=enc, down_block_additional_residuals=down,
                            mid_block_additional_residual=mid).sample
                loss = F.mse_loss(pred.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(f"step {step:05d}/{a.steps}  loss {loss.item():.4f}", flush=True)
            step += 1

    controlnet.save_pretrained(str(Path(a.out) / "model"))
    print(f"LoRA adapter saved to {Path(a.out) / 'model'}")


if __name__ == "__main__":
    main()
