# Choroid-Plexus-Segmentation
Repository containing my codes used to train, validate, and infer choroid plexus segmentation on T1 and FLAIR 3T MRI

## Data structure expected for preprocessing code to create the Torchio subjects to train faster
```bash
root
├── data
│ ├── train_subject_infos.csv
│ ├── S01
│ │   └── V01
│ │       ├── S01_V01_flair.nii.gz
│ │       └── S01_V01_mask.nii.gz
│ ├── S02
│ │   └── V01
│ │       ├── S02_V01_flair.nii.gz
│ │       └── S02_V01_mask.nii.gz
│ ├── ...
├── train_tio_subjects (created by preprocessing code)
│ ├── S01-V01.tio
│ ├── S02-V01.tio
│ ├── ...
```

## Citing us
-Yazdan-Panah A, Schmidt-Mengin M, Ricigliano VAG, Soulier T, Stankoff B, Colliot O. *Automatic segmentation of the choroid plexuses: Method and validation in controls and patients with multiple sclerosis.* Neuroimage Clin. 2023
[doi:10.1016/j.nicl.2023.103368](https://doi.org/10.1016/j.nicl.2023.103368)
