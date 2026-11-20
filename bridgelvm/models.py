"""BridgeLVM models (ViT / SegFormer with LoRA) and the retrained reference models.

Every reference model takes `pixel_values` and returns an HF-style output with `.logits`, so the same
trainers, losses and metrics are used for all models. Reference segmentation models return logits at
input resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _labels(cfg):
    id2label = dict(enumerate(cfg["labels"]))
    return id2label, {v: k for k, v in id2label.items()}


def _lora_config(cfg):
    from peft import LoraConfig
    return LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], bias="none",
                      target_modules=cfg["target_modules"], modules_to_save=cfg.get("modules_to_save", ["classifier"]))


def base_model(cfg):
    """Pretrained backbone with a new output layer (frozen weights are wrapped by LoRA afterwards)."""
    id2label, label2id = _labels(cfg)
    if cfg["family"] == "cls":
        from transformers import ViTForImageClassification as M
    else:
        from transformers import SegformerForSemanticSegmentation as M
    return M.from_pretrained(cfg["model_id"], num_labels=len(id2label), id2label=id2label, label2id=label2id,
                             ignore_mismatched_sizes=True)


def build_model(cfg):
    """Model to train for a recipe (configs.get)."""
    n = len(cfg["labels"])
    if cfg["arm"] == "vgg16":
        return VGG16Classifier(n)
    if cfg["arm"] == "alexnet":
        return AlexNetClassifier(n)
    if cfg["arm"] == "deepcrack":
        return DeepCrackNet(n)
    if cfg["arm"] == "deeplabv3p":
        return DeepLabV3Plus(n)
    from peft import get_peft_model
    model = get_peft_model(base_model(cfg), _lora_config(cfg))
    model.print_trainable_parameters()
    return model


def load_trained(cfg, model_dir):
    """Backbone + released LoRA adapter (+ SegFormer decode-head BatchNorm statistics)."""
    from pathlib import Path
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model(cfg), str(model_dir))
    bn = Path(model_dir) / "bn_stats.pt"
    if bn.exists():
        # The decode-head BatchNorm running statistics adapt to the target data during training but are not
        # part of a LoRA adapter, so they are stored next to it.
        missing = model.load_state_dict(torch.load(bn, map_location="cpu"), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    return model


# ---------------------------------------------------------------------------
# Reference models
# ---------------------------------------------------------------------------
def _cls_out(logits, labels):
    from transformers.modeling_outputs import ImageClassifierOutput
    loss = F.cross_entropy(logits, labels) if labels is not None else None
    return ImageClassifierOutput(loss=loss, logits=logits)


def _seg_out(logits):
    from transformers.modeling_outputs import SemanticSegmenterOutput
    return SemanticSegmenterOutput(logits=logits)


class VGG16Classifier(nn.Module):
    """ImageNet-pretrained VGG-16 (configuration D) with a new output layer."""

    def __init__(self, num_labels):
        super().__init__()
        import torchvision
        self.net = torchvision.models.vgg16(weights="IMAGENET1K_V1")
        self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, num_labels)

    def forward(self, pixel_values, labels=None):
        return _cls_out(self.net(pixel_values), labels)


class AlexNetClassifier(nn.Module):
    """ImageNet-pretrained AlexNet with a new output layer."""

    def __init__(self, num_labels):
        super().__init__()
        import torchvision
        self.net = torchvision.models.alexnet(weights="IMAGENET1K_V1")
        self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, num_labels)

    def forward(self, pixel_values, labels=None):
        return _cls_out(self.net(pixel_values), labels)


def _conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DeepCrackNet(nn.Module):
    """DeepCrack-BN (Liu et al., 2019): the 13 conv layers of VGG-16 with BatchNorm (ImageNet initialisation),
    a 1x1 side output after each of the 5 stages, upsampled to the input size and fused by a 1x1 conv.
    During training the side outputs are kept for deep supervision (see train_seg.add_deep_supervision)."""

    def __init__(self, num_labels=2):
        super().__init__()
        import torchvision
        feats = torchvision.models.vgg16_bn(weights="IMAGENET1K_V1").features
        self.stages, cur = nn.ModuleList(), []
        for m in feats:
            if isinstance(m, nn.MaxPool2d):          # stage boundary; pool5 is dropped
                self.stages.append(nn.Sequential(*cur))
                cur = []
            else:
                cur.append(m)
        self.pool = nn.MaxPool2d(2, 2)
        self.side = nn.ModuleList(nn.Conv2d(c, num_labels, 1) for c in (64, 128, 256, 512, 512))
        self.fuse = nn.Conv2d(5 * num_labels, num_labels, 1)

    def forward(self, pixel_values, labels=None):
        h, w = pixel_values.shape[-2:]
        x, sides = pixel_values, []
        for k, stage in enumerate(self.stages):
            if k > 0:
                x = self.pool(x)
            x = stage(x)
            sides.append(F.interpolate(self.side[k](x), size=(h, w), mode="bilinear", align_corners=False))
        out = _seg_out(self.fuse(torch.cat(sides, dim=1)))
        if self.training:
            self._side_logits = sides
        return out


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ (Chen et al., 2018): ResNet-50 encoder (ImageNet), output stride 16, ASPP (rates 6/12/18),
    decoder with a 48-channel low-level projection."""

    def __init__(self, num_labels=2):
        super().__init__()
        import torchvision
        from torchvision.models.segmentation.deeplabv3 import ASPP
        r = torchvision.models.resnet50(weights="IMAGENET1K_V2", replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.aspp = ASPP(2048, [6, 12, 18], out_channels=256)
        self.low = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.head = nn.Sequential(_conv_bn_relu(256 + 48, 256), _conv_bn_relu(256, 256), nn.Conv2d(256, num_labels, 1))

    def forward(self, pixel_values, labels=None):
        h, w = pixel_values.shape[-2:]
        low = self.layer1(self.stem(pixel_values))
        x = self.aspp(self.layer4(self.layer3(self.layer2(low))))
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.head(torch.cat([x, self.low(low)], dim=1))
        return _seg_out(F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False))
