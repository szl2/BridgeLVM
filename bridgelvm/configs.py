"""Training recipes for every task (BridgeLVM) and for the retrained reference models.

Batch sizes are per optimisation step on one GPU.
"""

VIT = "google/vit-base-patch16-224-in21k"
SEGFORMER = "nvidia/segformer-b0-finetuned-ade-512-512"

# Folder under data/ used by each task
DATASET_DIR = {
    "t1.1": "SDNET2018",
    "t1.2": "BearingCondition",
    "t1.3": "CODEBRIM",
    "t2.1": "DeepCrack",
    "t2.2": "CorrosionCondition",
}

_QKV = ["query", "key", "value"]

TASKS = {
    # T1.1 SDNET2018: crack / no crack
    "t1.1": dict(family="cls", labels=["no_crack", "crack"], model_id=VIT,
                 lora_r=32, lora_alpha=64, lora_dropout=0.05, target_modules=_QKV,
                 lr=2.5e-4, batch=64, epochs=10, weight_decay=0.01, warmup=0.1,
                 trainer="plain", transforms="default", select_metric="accuracy"),
    # T1.2 BearingCondition (300x300): Good / Fair / Poor / Severe
    "t1.2": dict(family="cls", labels=["Good", "Fair", "Poor", "Severe"], model_id=VIT,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05, target_modules=_QKV,
                 lr=5e-4, batch=32, epochs=40, weight_decay=0.01, warmup=0.1,
                 trainer="weighted", scheduler="cosine", num_cycles=1, weight_power=0.5, label_smoothing=0.05,
                 transforms="strong", select_metric="f1_macro"),
    # T1.3 CODEBRIM: spallation / no spallation
    "t1.3": dict(family="cls", labels=["NoSpallation", "Spallation"], model_id=VIT,
                 lora_r=32, lora_alpha=64, lora_dropout=0.05, target_modules=_QKV,
                 lr=2.5e-4, batch=32, epochs=30, weight_decay=0.01, warmup=0.1,
                 trainer="weighted", scheduler="cosine", num_cycles=1, weight_power=0.5, label_smoothing=0.05,
                 transforms="strong", select_metric="f1_macro"),
    # T2.1 DeepCrack: crack segmentation
    "t2.1": dict(family="seg", labels=["background", "crack"], model_id=SEGFORMER,
                 lora_r=32, lora_alpha=64, lora_dropout=0.05, target_modules=_QKV, modules_to_save=["classifier"],
                 lr=1.2e-4, batch=16, epochs=50, weight_decay=0.01, warmup=0.05, scheduler="cosine",
                 metrics="crack", select_metric="mean_iou"),
    # T2.2 CorrosionCondition: corrosion segmentation (any corrosion vs background)
    "t2.2": dict(family="seg", labels=["background", "corrosion"], model_id=SEGFORMER,
                 lora_r=32, lora_alpha=64, lora_dropout=0.05, target_modules=_QKV, modules_to_save=["classifier"],
                 lr=1.2e-4, batch=16, epochs=50, weight_decay=0.01, warmup=0.05, scheduler="cosine",
                 metrics="corrosion", select_metric="mean_iou"),
}

# Reference models retrained on the same splits (all parameters trained, ImageNet initialisation).
# Each uses the training recipe of its task, except for the settings listed here.
REFERENCES = {
    "vgg16": dict(task="t1.3", lr=1e-4),                       # VGG-16 (configuration D)
    "deepcrack": dict(task="t2.1", lr=9e-4),                   # DeepCrack (VGG-16-BN, deep supervision)
    "deeplabv3p": dict(task="t2.2", lr=9e-4),                  # DeepLabV3+ (ResNet-50)
    # AlexNet uses its own training loop (scripts/train.py: train_alexnet)
    "alexnet": dict(task="t1.1", lr=1e-4, batch=64, epochs=20, weight_decay=0.01, warmup=0.1),
}

# T3.1 deterioration-state synthesis (Stable Diffusion v1.5 + ControlNet, LoRA on the ControlNet)
SYNTHESIS = dict(
    base_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet="lllyasviel/control_v11p_sd15_seg",
    lora_r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lr=1e-4, batch=4, steps=3000, resolution=512,
    delta_text="+12 months corrosion progression",
    strength=0.55, guidance_scale=7.5, num_inference_steps=30,
)


def get(task, arm="lora"):
    """Recipe for a task; for a reference arm, the task recipe with the reference overrides."""
    if arm == "lora":
        return dict(TASKS[task], task=task, arm=arm)
    ref = REFERENCES[arm]
    if ref["task"] != task:
        raise ValueError(f"--arm {arm} is defined for task {ref['task']}, not {task}")
    cfg = dict(TASKS[task], task=task, arm=arm)
    cfg.update({k: v for k, v in ref.items() if k != "task"})
    return cfg
