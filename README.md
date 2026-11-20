# BridgeLVM

Code and trained models for **BridgeLVM: A Study of Parameter-Efficient Foundation Model Adaptation for Infrastructure Inspection**. A single adaptation strategy, LoRA, is applied to three task types: classification with ViT-Base/16, segmentation with SegFormer-B0, and deterioration-state synthesis with Stable Diffusion v1.5 + ControlNet. It is evaluated on five public bridge inspection datasets.

## Installation
```bash
git clone https://github.com/szl2/BridgeLVM.git
cd BridgeLVM
conda create -n bridgelvm python=3.11 -y
conda activate bridgelvm
pip install -r requirements.txt
```
Tested with PyTorch 2.9.0 (CUDA 12.8) on an NVIDIA RTX 3090 (24 GB). Run all commands from the repository root.

## Datasets
| Task | Dataset | Download |
|---|---|---|
| T1.1 Crack classification | SDNET2018 | https://doi.org/10.15142/T3TD19 |
| T1.2 Bearing condition | BearingCondition | https://doi.org/10.7294/16624642.v1 |
| T1.3 Spallation classification | CODEBRIM | https://doi.org/10.5281/zenodo.2620293\* |
| T2.1 Crack segmentation | DeepCrack | https://github.com/yhlleo/DeepCrack |
| T2.2 Corrosion segmentation | CorrosionCondition | https://doi.org/10.7294/16624663.v2 |
| T3.1 Deterioration-state synthesis | CorrosionCondition | same as T2.2 |

\* Alternative: https://universe.roboflow.com/defect-detection-edbnh/codebrim-lnsfg

Put each dataset into its folder under `data/` (layouts: [`data/README.md`](data/README.md)):
```bash
mkdir -p downloads
# DeepCrack
curl -L -o downloads/DeepCrack.zip https://github.com/yhlleo/DeepCrack/raw/master/dataset/DeepCrack.zip
unzip -q downloads/DeepCrack.zip -d data/DeepCrack
# CorrosionCondition
curl -L -o downloads/corrosion.zip https://ndownloader.figshare.com/files/31729733
unzip -q downloads/corrosion.zip -d downloads
cp -r "downloads/Corrosion Condition State Classification/512x512/"* data/CorrosionCondition/
# BearingCondition
curl -L -o downloads/bearing.zip https://ndownloader.figshare.com/files/30914398
unzip -q downloads/bearing.zip -d downloads
cp -r "downloads/Bearing Condition State Classifier/300x300/"* data/BearingCondition/
```
- SDNET2018: download in a browser and unzip into `data/SDNET2018/` (folders `D`, `P`, `W`).
- CODEBRIM: download the original images in a browser, unzip, then run `python scripts/prepare_codebrim.py --source <unzipped folder>`.

The train/validation/test splits used in the paper are fixed in `splits/`.

## Trained models
```bash
curl -LO https://github.com/szl2/BridgeLVM/releases/download/v1.0/bridgelvm_models_v1.zip
unzip -q bridgelvm_models_v1.zip -d models/
```

| Task | Backbone | LoRA rank | Size |
|---|---|---|---|
| T1.1 | ViT-Base/16 | 32 | 6.8 MB |
| T1.2 | ViT-Base/16 | 16 | 3.5 MB |
| T1.3 | ViT-Base/16 | 32 | 6.8 MB |
| T2.1 | SegFormer-B0 | 32 | 0.8 MB |
| T2.2 | SegFormer-B0 | 32 | 0.8 MB |
| T3.1 | ControlNet (SD v1.5) | 8 | 2.9 MB |

## Test the code
```bash
python scripts/evaluate.py --task t1.1 --model models/t1.1
python scripts/evaluate.py --task t1.2 --model models/t1.2
python scripts/evaluate.py --task t1.3 --model models/t1.3
python scripts/evaluate.py --task t2.1 --model models/t2.1
python scripts/evaluate.py --task t2.2 --model models/t2.2
```

| Task | Metric | Test score (%) |
|---|---|---|
| T1.1 | Accuracy | 93.96 |
| T1.2 | Accuracy | 77.55 |
| T1.3 | Accuracy | 80.19 |
| T2.1 | mIoU | 85.74 |
| T2.2 | Macro F1 | 86.51 |

T1.3 uses images prepared from the Zenodo release; with the Roboflow images it is 83.02%.

Deterioration-state images (T3.1) for the 44 CorrosionCondition test images:
```bash
python scripts/synthesize.py --model models/t3.1 --out outputs/t3.1
```

## Train
BridgeLVM (LoRA):
```bash
python scripts/train.py --task t2.1 --out runs/t2.1
python scripts/train_synthesis.py --out runs/t3.1
```
Reference models, retrained on the same splits:
```bash
python scripts/train.py --task t1.1 --arm alexnet    --out runs/ref_t1.1
python scripts/train.py --task t1.3 --arm vgg16      --out runs/ref_t1.3
python scripts/train.py --task t2.1 --arm deepcrack  --out runs/ref_t2.1
python scripts/train.py --task t2.2 --arm deeplabv3p --out runs/ref_t2.2
```
Released BearingCondition reference model (https://doi.org/10.7294/16628698.v1, expected accuracy 78.95%):
```bash
curl -L -o downloads/bearing_model.zip https://ndownloader.figshare.com/files/30914440
unzip -q downloads/bearing_model.zip -d downloads
python scripts/eval_bearing_reference.py --checkpoint downloads/*/l2_loss/checkpoint_9.pt
```
All hyperparameters are in `bridgelvm/configs.py`.

## Citation
```bibtex
@article{bridgelvm,
  title   = {BridgeLVM: A Study of Parameter-Efficient Foundation Model Adaptation for Infrastructure Inspection},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

## License
Code: MIT. Each dataset and the BearingCondition reference model are subject to their own licenses.
