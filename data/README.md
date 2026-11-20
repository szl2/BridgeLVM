# Datasets

Download each dataset from its official source and place it in the folder named below. The train / validation / test splits used in the paper are fixed in [`../splits`](../splits), so no further splitting is needed.

## SDNET2018 (T1.1)
1. Download from https://doi.org/10.15142/T3TD19.
2. Unzip it so that the folders `D`, `P` and `W` are directly inside `data/SDNET2018/`.

```
data/SDNET2018/
├── D/{CD,UD}/*.jpg
├── P/{CP,UP}/*.jpg
└── W/{CW,UW}/*.jpg
```

## BearingCondition (T1.2)
1. Download `Bearing Condition State Classifier.zip` from https://doi.org/10.7294/16624642.v1.
2. Copy the two folders inside `300x300/` into `data/BearingCondition/`.

```
data/BearingCondition/
├── Train_300x300/{1,2,3,4}/*.jpeg
└── Test_300x300/{1,2,3,4}/*.jpeg
```

To evaluate the released BearingCondition reference model, also download it from https://doi.org/10.7294/16628698.v1 and unzip it anywhere.

## CODEBRIM (T1.3)
1. Download the original images from https://doi.org/10.5281/zenodo.2620293 and unzip them anywhere. The unzipped folder contains `images/` and `annotations/`.
   - Alternative source: the Roboflow export at https://universe.roboflow.com/defect-detection-edbnh/codebrim-lnsfg (format: Multi-Label Classification).
2. Run the following command. It writes the 1,052 annotated images, resized to 640×640, to `data/CODEBRIM/images/`.

   ```bash
   python scripts/prepare_codebrim.py --source <unzipped CODEBRIM folder>
   ```

```
data/CODEBRIM/
└── images/image_*.jpg
```

## DeepCrack (T2.1)
1. Download `dataset/DeepCrack.zip` from https://github.com/yhlleo/DeepCrack.
2. Unzip it into `data/DeepCrack/`.

```
data/DeepCrack/
├── train_img/*.jpg   train_lab/*.png
└── test_img/*.jpg    test_lab/*.png
```

## CorrosionCondition (T2.2, T3.1)
1. Download `Corrosion Condition State Classification.zip` from https://doi.org/10.7294/16624663.v2.
2. Copy the two folders inside `512x512/` into `data/CorrosionCondition/`.

```
data/CorrosionCondition/
├── Train/{images_512,mask_512}/
└── Test/{images_512,mask_512}/
```
