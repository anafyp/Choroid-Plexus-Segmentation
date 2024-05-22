#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:08:48 2022

@author: arya.yazdan-panah
"""
import os
import pickle
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchio as tio
import numpy as np
import pandas as pd
import wandb
from utils import UNet3d
from losses import ComboLoss

# =============================================================================
# Directory definition + Configs
# =============================================================================
np.seterr(under='warn')
root = "..."
data_dir = os.path.join(root, "data", "train_torch")
train_dir = os.path.join(root, "trainings")
code_dir = os.path.join(root, "codes")
criterion = ComboLoss()
log = True
resume = True

config = dict(batch_size=4,
              patch_size=48,
              num_patches_per_image=12,
              inference_max_patches=500,
              positive_threshold=0.8,
              low_res_shape=[72, 104, 104],
              n_stages_1=4,
              n_stages_2=4,
              norm="gn4",
              non_lin="leaky_relu",
              num_init_filters_1=16,
              num_init_filters_2=16,
              n_epochs=400,
              initial_learning_rate=1e-3,
              scheduler="ReduceLROnPlateau",
              augmentation=True,
              loss="ComboLoss")
logging_dir = os.path.join(train_dir, "2step_old", f"{config['augmentation']}Augm")
if not os.path.isdir(logging_dir):
    os.makedirs(logging_dir)
print("Directories defined")
# =============================================================================
# Augmentation
# =============================================================================
# Spatial Transforms
canonical = tio.ToCanonical(p=1)
resample = tio.Resample(target=1, image_interpolation='linear', p=1)
crop_or_pad = tio.CropOrPad(target_shape=(176, 256, 256), p=1)
random_flip = tio.RandomFlip(axes=("LR",), p=0.5, include=["gth", "image"])
random_affine = tio.RandomAffine(scales=0.3, degrees=15, default_pad_value=0, p=0.3, include=["gth", "image"])
random_anisotropy = tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=2, p=0.3, include=["gth", "image"])
# elastic defformation, ecrire les images
# random_ED = tio.RandomElasticDeformation(num_control_points=, max_displacement, locked_borders=, image_interpolation=, p=1)
# Intensity transforms
random_motion = tio.RandomMotion(degrees=15, translation=15, num_transforms=2, p=0.3, exclude=["gth"])
random_ghosting = tio.RandomGhosting(num_ghosts=2, axes=(0, 1, 2), intensity=(0.5, 1), restore=0.02, p=0.3, exclude=["gth"])
random_BF = tio.RandomBiasField(coefficients=0.5, order=3, p=0.3, exclude=["gth"])
random_spike = tio.RandomSpike(num_spikes=1, intensity=(1, 3), p=0.3, exclude=["gth"])
random_noise = tio.RandomNoise(mean=0, std=(0, 0.25), p=0.3, exclude=["gth"])
rescaling = tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(0, 100), p=1, exclude=["gth"])

validation_transforms = tio.Compose([])
training_transforms = tio.Compose([random_flip,
                                   random_affine,
                                   random_anisotropy,
                                   random_motion,
                                   random_ghosting,
                                   random_BF,
                                   random_spike,
                                   random_noise,
                                   rescaling])
print("Augmentations defined")
# =============================================================================
# Data Loading
# =============================================================================
dataset_infos = os.path.join(root, "data", "train_subject_infos.csv")
info_df = pd.read_csv(dataset_infos)
subjects = sorted(glob.glob(os.path.join(data_dir, "*.tio")))
subjects = [torch.load(s) for s in subjects]
subjects_train = tio.SubjectsDataset(subjects, training_transforms)
subjects_validation = tio.SubjectsDataset(subjects, validation_transforms)
with open(os.path.join(data_dir, "folds.pickle"), "rb") as f:
    folds = pickle.load(f)
print("Data Loaded")
# =============================================================================
# Training
# =============================================================================
for f, fold in enumerate(folds):
    if f != 4:
        continue
    # Training dataloader
    train_idx = []
    for name in fold["train"]:
        train_idx.extend(info_df.index[info_df["SUBJECT"] == name].tolist())
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    training_loader = DataLoader(subjects_train, batch_size=config["batch_size"], sampler=train_subsampler)
    # Validation dataloader
    val_idx = []
    for name in fold["val"]:
        val_idx.extend(info_df.index[info_df["SUBJECT"] == name].tolist())
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    validation_loader = DataLoader(subjects_validation, batch_size=1, sampler=val_subsampler)
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    stage_1 = UNet3d(in_channels=1,
                     out_channels=1,
                     num_stages=config['n_stages_1'],
                     initial_num_channels=config['num_init_filters_1'],
                     norm=config['norm'],
                     non_lin=config['non_lin'],
                     kernel_size=3,
                     pooling='avg').to(device)
    stage_2 = UNet3d(in_channels=1,
                     out_channels=1,
                     num_stages=config['n_stages_2'],
                     initial_num_channels=config['num_init_filters_2'],
                     norm=config['norm'],
                     non_lin=config['non_lin'],
                     kernel_size=3,
                     pooling='avg').to(device)
    optimizer = torch.optim.Adam(list(stage_1.parameters()) + list(stage_2.parameters()), config["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=50,
                                                           threshold=0.0001,
                                                           threshold_mode='rel',
                                                           cooldown=10,
                                                           min_lr=0,
                                                           eps=1e-08, verbose=False)
    # Fold dir
    fold_dir = os.path.join(logging_dir, "fold_" + str(f + 1))
    if not os.path.isdir(fold_dir):
        os.makedirs(fold_dir)
    # Start Logging
    if log:
        run = wandb.init(project="",
                         entity="",
                         config=config,
                         save_code=True,
                         group="",
                         mode="online",
                         name="Fold" + str(f + 1),
                         dir=logging_dir,
                         reinit=True)
    for e in range(config["n_epochs"]):
        e_path = os.path.join(fold_dir, "model_epoch_" + str(e + 1) + ".pt")
        if os.path.isfile(e_path) and resume:
            print('skipping fold:', f, 'epoch:', e)
            continue
        if not(os.path.isfile(e_path)) and resume:
            to_load = os.path.join(fold_dir, "model_epoch_" + str(e) + ".pt")
            if os.path.isfile(to_load) and resume:
                print(to_load)
                checkpoint = torch.load(to_load, map_location=device)
                stage_1.load_state_dict(checkpoint['stage_1_state_dict'])
                stage_2.load_state_dict(checkpoint['stage_2_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        stage_1_train_losses, stage_1_val_losses = [], []
        stage_1_train_bces, stage_1_val_bces = [], []
        stage_1_train_dices, stage_1_val_dices = [], []
        stage_2_train_losses, stage_2_val_losses = [], []
        stage_2_train_bces, stage_2_val_bces = [], []
        stage_2_train_dices, stage_2_val_dices = [], []
        train_losses, val_losses = [], []
        num_above_thr_train, num_above_thr_val = [], []
        with torch.enable_grad():
            stage_1.train()
            stage_2.train()
            for i, batch in enumerate(training_loader):
                optimizer.zero_grad()
                x_highres, y_highres = (batch["image"]["data"].to(device),
                                        batch["gth"]["data"].float().to(device))
                x_lowres = F.interpolate(x_highres, config["low_res_shape"], align_corners=True, mode='trilinear')
                y_lowres = F.interpolate(y_highres, config["low_res_shape"], align_corners=True, mode='trilinear')
                b, c, d, h, w = x_lowres.shape
                b, c, D, H, W = x_highres.shape
                stage_1_seg = stage_1(x_lowres).sigmoid()
                stage_1_train_dice, stage_1_train_bce, stage_1_train_loss = criterion(stage_1_seg, y_lowres)
                train_loss = stage_1_train_loss
                stage_1_train_bce = stage_1_train_bce.detach().cpu().numpy()
                stage_1_train_dice = stage_1_train_dice.detach().cpu().numpy()
                stage_1_train_loss = (stage_1_train_bce + stage_1_train_dice) / 2
                padding = config['patch_size'] // 2
                x_highres_padded = F.pad(x_highres, (padding, padding, padding, padding, padding, padding))
                y_highres_padded = F.pad(y_highres, (padding, padding, padding, padding, padding, padding))
                patches_x = []
                patches_y = []
                num_above_threshold = 0
                for i in range(b):
                    above_threshold_ind = torch.nonzero((stage_1_seg[i][0] >= config['positive_threshold']).int())
                    above_threshold_ind = above_threshold_ind[torch.randperm(above_threshold_ind.shape[0])]
                    num_above_threshold += above_threshold_ind.shape[0]
                    for j in range(min(config["num_patches_per_image"], above_threshold_ind.shape[0])):
                        loc_lowres = above_threshold_ind[j].float()
                        loc_highres = loc_lowres / torch.tensor([d, h, w], device=device) * torch.tensor([D, H, W], device=device)
                        bound_min = (loc_highres - config['patch_size'] / 2.).round().int() + padding
                        bound_max = bound_min + config['patch_size']
                        selection = (i, slice(None, None, None)) + tuple([slice(bound_min[k].item(), bound_max[k].item()) for k in range(3)])
                        patches_x.append(x_highres_padded[selection])
                        patches_y.append(y_highres_padded[selection])
                num_above_thr_train.append(num_above_threshold / (i + 1))
                if not num_above_threshold:
                    stage_2_train_loss, stage_2_train_bce, stage_2_train_dice = np.nan, np.nan, np.nan
                else:
                    patches_x = torch.stack(patches_x, 0)
                    patches_y = torch.stack(patches_y, 0)
                    stage_2_train_dice, stage_2_train_bce, stage_2_train_loss = criterion(stage_2(patches_x).sigmoid(), patches_y)
                    train_loss = train_loss + stage_2_train_loss
                    stage_2_train_bce = stage_2_train_bce.detach().cpu().numpy()
                    stage_2_train_dice = stage_2_train_dice.detach().cpu().numpy()
                    stage_2_train_loss = (stage_2_train_bce + stage_2_train_dice) / 2
                stage_1_train_losses.append(stage_1_train_loss)
                stage_1_train_bces.append(stage_1_train_bce)
                stage_1_train_dices.append(stage_1_train_dice)
                stage_2_train_losses.append(stage_2_train_loss)
                stage_2_train_bces.append(stage_2_train_bce)
                stage_2_train_dices.append(stage_2_train_dice)
                train_losses.append(train_loss.detach().cpu().numpy())
                train_loss.backward()
                optimizer.step()
        with torch.no_grad():
            stage_1.eval()
            stage_2.eval()
            for i, batch in enumerate(validation_loader):
                x, y = (batch["image"]["data"].to(device),
                        batch["gth"]["data"].float().to(device))
                x_lowres = F.interpolate(x, config["low_res_shape"], align_corners=True, mode='trilinear')
                y_lowres = F.interpolate(y, config["low_res_shape"], align_corners=True, mode='trilinear')
                b, c, d, h, w = x_lowres.shape
                b, c, D, H, W = x.shape
                stage_1_seg = stage_1(x_lowres).sigmoid()
                stage_1_val_dice, stage_1_val_bce, stage_1_val_loss = criterion(stage_1_seg, y_lowres)
                val_loss = stage_1_val_loss
                stage_1_val_bce = stage_1_val_bce.detach().cpu().numpy()
                stage_1_val_dice = stage_1_val_dice.detach().cpu().numpy()
                stage_1_val_loss = (stage_1_val_bce + stage_1_val_dice) / 2
                patch_size = config['patch_size']
                padding = patch_size // 2
                x_highres_padded = F.pad(x, (padding, padding, padding, padding, padding, padding))
                patches_x = []
                num_above_threshold = 0
                selections = []
                for i in range(b):
                    above_threshold_ind = torch.nonzero((stage_1_seg[i][0] >= config['positive_threshold']).int())
                    above_threshold_ind = above_threshold_ind[torch.randperm(above_threshold_ind.shape[0])]
                    num_above_threshold += above_threshold_ind.shape[0]
                    for j in range(min(config["inference_max_patches"], above_threshold_ind.shape[0])):
                        loc_lowres = above_threshold_ind[j].float().to(device)
                        loc_highres = loc_lowres / torch.tensor([d, h, w], device=device) * torch.tensor([D, H, W], device=device)
                        bound_min = (loc_highres - config['patch_size'] / 2.).round().int() + padding
                        bound_max = bound_min + config['patch_size']
                        selection = (i, slice(None, None, None)) + tuple([slice(bound_min[k].item(), bound_max[k].item()) for k in range(3)])
                        selections.append(selection)
                        patches_x.append(x_highres_padded[selection])
                num_above_thr_val.append(num_above_threshold / (i + 1))
                if not num_above_threshold:
                    stage_2_val_loss, stage_2_val_bce, stage_2_val_dice = np.nan, np.nan, np.nan
                else:
                    patches_x = torch.stack(patches_x, 0)
                    loader = torch.utils.data.DataLoader(patches_x, batch_size=64, shuffle=False, drop_last=False, num_workers=0)
                    predictions = []
                    for i, batch in enumerate(loader):
                        predictions.append(stage_2(batch))
                    predictions = torch.cat(predictions, 0).sigmoid()
                    soft_overlap = True
                    if soft_overlap:
                        mask_1d = torch.linspace(0, 2 * np.pi, patch_size)
                        mask_1d = 1 - mask_1d.cos()
                        A, B, C = torch.meshgrid(mask_1d, mask_1d, mask_1d)
                        mask = A * B * C
                    else:
                        mask = torch.ones(patch_size, patch_size, patch_size)
                    y_pred = torch.zeros(*x_highres_padded.shape).to(device)
                    y_norm = torch.zeros(*x_highres_padded.shape).to(device)
                    mask = mask.to(device)
                    for s, patch in zip(selections, predictions):
                        y_pred[s] += patch[0] * mask
                        y_norm[s] += mask
                    output_full = y_pred / y_norm
                    output_full[torch.isnan(output_full)] = 0
                    output_full = output_full[:, :, padding:-padding, padding:-padding, padding:-padding]
                    stage_2_val_dice, stage_2_val_bce, stage_2_val_loss = criterion(output_full, y)
                    val_loss = val_loss + stage_2_val_loss
                    stage_2_val_bce = stage_2_val_bce.detach().cpu().numpy()
                    stage_2_val_dice = stage_2_val_dice.detach().cpu().numpy()
                    stage_2_val_loss = (stage_2_val_bce + stage_2_val_dice) / 2
                stage_1_val_losses.append(stage_1_val_loss)
                stage_1_val_bces.append(stage_1_val_bce)
                stage_1_val_dices.append(stage_1_val_dice)
                stage_2_val_losses.append(stage_2_val_loss)
                stage_2_val_bces.append(stage_2_val_bce)
                stage_2_val_dices.append(stage_2_val_dice)
                val_losses.append(val_loss.detach().cpu().numpy())
        scheduler.step(val_loss)
        if log:
            run.log({"patches_above_thr_train": np.nanmean(np.array(num_above_thr_train)),
                     "patches_above_thr_val": np.nanmean(np.array(num_above_thr_val)),
                     "train_loss_stage1": np.nanmean(np.array(stage_1_train_losses)),
                     "train_bce_stage1": np.nanmean(np.array(stage_1_train_bces)),
                     "train_dice_stage1": np.nanmean(np.array(stage_1_train_dices)),
                     "train_loss_stage2": np.nanmean(np.array(stage_2_train_losses)),
                     "train_bce_stage2": np.nanmean(np.array(stage_2_train_bces)),
                     "train_dice_stage2": np.nanmean(np.array(stage_2_train_dices)),
                     "train_loss": np.nanmean(np.array(train_losses)),
                     "val_loss_stage1": np.nanmean(np.array(stage_1_val_losses)),
                     "val_bce_stage1": np.nanmean(np.array(stage_1_val_bces)),
                     "val_dice_stage1": np.nanmean(np.array(stage_1_val_dices)),
                     "val_loss_stage2": np.nanmean(np.array(stage_2_val_losses)),
                     "val_bce_stage2": np.nanmean(np.array(stage_2_val_bces)),
                     "val_dice_stage2": np.nanmean(np.array(stage_2_val_dices)),
                     "val_loss": np.nanmean(np.array(val_losses)),
                     "lr": optimizer.param_groups[0]["lr"]})
        torch.save({'stage_1_state_dict': stage_1.state_dict(),
                    'stage_2_state_dict': stage_2.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config}, e_path)
    run.finish()
