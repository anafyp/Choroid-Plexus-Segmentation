#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:49:31 2022

@author: arya.yazdan-panah
"""

import os
import torch
import torchio as tio
import pandas as pd

root_dir = "......"
dataset_infos = os.path.join(root_dir, "train_subject_infos.csv")
input_dir = os.path.join(root_dir, "data_nifti")
output_dir = os.path.join(root_dir, "data_torch")
info_df = pd.read_csv(dataset_infos)
# Transforms
canonical = tio.ToCanonical(p=1)
resample = tio.Resample(target=1, image_interpolation='linear', p=1)
crop_or_pad = tio.CropOrPad(target_shape=(176, 256, 256), p=1)
rescaling = tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(0.5, 99.5), p=1, exclude=["gth"])
transforms = tio.Compose([canonical, resample, crop_or_pad, rescaling])
transforms = tio.Compose([canonical])

# Converting Images to pytorch tensors
for i, subject_infos in info_df.iterrows():
    s = subject_infos["SUBJECT"]
    v = subject_infos["VISIT"]
    sdir = os.path.join(input_dir, s, v)
    image_path = os.path.join(sdir, "%s_%s_image.nii.gz" % (s, v))
    label_path = os.path.join(sdir, "%s_%s_mask.nii.gz " % (s, v))
    image = tio.ScalarImage(image_path)
    mask = tio.LabelMap(label_path)
    # creating tio subject object and preprocessing it
    subject = tio.Subject(image=image, gth=mask)
    subject = transforms(subject)
    out_tensor_path = os.path.join(output_dir, "-".join([subject_infos["SUBJECT"], subject_infos["VISIT"]]) + ".tio")
    torch.save(subject, out_tensor_path)
