#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio
import time
import os
import argparse
import sys


# UNet code from https://github.com/marius-sm/nd_unet
def get_norm(name, num_channels, dim=None):
    if name == 'bn':
        assert dim is not None, 'Please specify the dim argument (1, 2 or 3D)'
        if dim == 1:
            norm = nn.BatchNorm1d(num_channels)
        if dim == 2:
            norm = nn.BatchNorm2d(num_channels)
        if dim == 3:
            norm = nn.BatchNorm3d(num_channels)
        return norm
    elif 'gn' in name:
        num_groups = name[2:]
        if num_groups == '':
            num_groups = 8
        num_groups = int(num_groups)
        return nn.GroupNorm(num_groups, num_channels)
    elif name == 'in':
        return nn.GroupNorm(num_channels, num_channels)
    elif name == 'ln':
        return nn.GroupNorm(1, num_channels)
    else:
        raise ValueError(f"Normalization '{name}' not recognized. Possible values are None (no normalization), 'bn' (batch norm), 'gnx' (group norm where x is optionally the number of groups), 'in' (instance norm), 'ln' (layer norm)")


def get_non_lin(name):
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Activation {name} not recognized. Possible values are 'relu', 'leaky_relu', 'gelu', 'elu'")


def get_conv(dim, *args, **kwargs):
    if dim == 1:
        return nn.Conv1d(*args, **kwargs)
    if dim == 2:
        return nn.Conv2d(*args, **kwargs)
    if dim == 3:
        return nn.Conv3d(*args, **kwargs)


def get_conv_block(dim, in_channels, out_channels, norm, non_lin,
                   kernel_size=3):
    padding = kernel_size // 2
    layers = [get_conv(dim, in_channels=in_channels, out_channels=out_channels,
                       kernel_size=kernel_size, padding=padding)]
    if norm is not None:
        layers.append(get_norm(norm, num_channels=out_channels, dim=dim))
    if non_lin is not None:
        layers.append(get_non_lin(non_lin))
    return nn.Sequential(*layers)


class UNetEncoder(nn.Module):
    def __init__(self, dim, in_channels, num_stages, initial_num_channels,
                 norm=None, non_lin='relu', kernel_size=3, pooling='max'):
        super().__init__()
        assert pooling in ['avg', 'max'], "pooling can be 'avg' or 'max'"
        if dim == 1:
            if pooling == 'avg':
                self.pooling = nn.AvgPool1d(2, 2)
            else:
                self.pooling = nn.MaxPool1d(2, 2)
        if dim == 2:
            if pooling == 'avg':
                self.pooling = nn.AvgPool2d(2, 2)
            else:
                self.pooling = nn.MaxPool2d(2, 2)
        if dim == 3:
            if pooling == 'avg':
                self.pooling = nn.AvgPool3d(2, 2)
            else:
                self.pooling = nn.MaxPool3d(2, 2)
        self.module_list = nn.ModuleList()
        for i in range(num_stages):
            block_1_in_channels = in_channels if i == 0 else (2 ** i) * initial_num_channels
            block_1_out_channels = (2 ** i) * initial_num_channels
            block_2_in_channels = block_1_out_channels
            block_2_out_channels = (2 ** (i + 1)) * initial_num_channels
            m = nn.Sequential(get_conv_block(dim=dim, in_channels=block_1_in_channels, out_channels=block_1_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin),
                              get_conv_block(dim=dim, in_channels=block_2_in_channels, out_channels=block_2_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin))
            self.module_list.append(m)

    def forward(self, x):
        acts = []
        for m in self.module_list[:-1]:
            x = m(x)
            acts.append(x)
            x = self.pooling(x)
        x = self.module_list[-1](x)
        return x, acts


class UNetDecoder(nn.Module):
    def __init__(self, dim, out_channels, num_stages, initial_num_channels, norm=None, non_lin='relu', kernel_size=3):
        super().__init__()
        self.module_list = nn.ModuleList()
        for i in range(num_stages - 1):
            block_in_channels = (2 ** (i + 1) + (2 ** (i + 2))) * initial_num_channels
            block_out_channels = (2 ** (i + 1)) * initial_num_channels
            m = nn.Sequential(get_conv_block(dim=dim, in_channels=block_in_channels, out_channels=block_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin),
                              get_conv_block(dim=dim, in_channels=block_out_channels, out_channels=block_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin))
            self.module_list.append(m)
        self.final_conv = get_conv(dim, 2 * initial_num_channels, out_channels, 1, padding=0)

    def forward(self, x, acts):
        interpolation = 'linear'
        if x.dim() == 4:
            interpolation = 'bilinear'
        if x.dim() == 5:
            interpolation = 'trilinear'
        for y, m in zip(reversed(acts), reversed(self.module_list)):
            x = F.interpolate(x, y.shape[2:], mode=interpolation, align_corners=True)
            x = m(torch.cat([y, x], 1))
        x = self.final_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, dim, in_channels, out_channels, num_stages, initial_num_channels, norm=None, non_lin='relu', kernel_size=3, pooling='max'):
        super().__init__()

        self.encoder = UNetEncoder(dim, in_channels, num_stages, initial_num_channels, norm=norm, non_lin=non_lin, kernel_size=kernel_size, pooling=pooling)
        self.decoder = UNetDecoder(dim, out_channels, num_stages, initial_num_channels, norm=norm, non_lin=non_lin, kernel_size=kernel_size)

    def forward(self, x):

        x, acts = self.encoder(x)
        x = self.decoder(x, acts)

        return x


def UNet1d(*args, **kwargs):
    return UNet(1, *args, **kwargs)


def UNet2d(*args, **kwargs):
    return UNet(2, *args, **kwargs)


def UNet3d(*args, **kwargs):
    return UNet(3, *args, **kwargs)


@torch.no_grad()
def inference(x_highres, config, stage_1, stage_2, soft_overlap=True):
    # x_highres should have shape (c, D, H, W), i.e. no batch dimension
    low_res_shape = config['low_res_shape']
    x_lowres = F.interpolate(x_highres[None, ...].to(device), low_res_shape, align_corners=True, mode='trilinear')[0]
    c, d, h, w = x_lowres.shape
    c, D, H, W = x_highres.shape
    print('Running stage 1...')
    t0 = time.time()
    stage_1_output = stage_1(x_lowres[None, ...])[0]
    print(f'Done in {(time.time()-t0):.4f} seconds.')
    stage_1_logits = stage_1_output[0, :, :, :]  # has shape (d, h, w)
    stage_1_seg = stage_1_logits.sigmoid()  # has shape (d, h, w)
    positive_threshold = config['positive_threshold']
    above_threshold = (stage_1_seg >= positive_threshold).int()  # has shape (d, h, w)
    above_threshold_ind = torch.nonzero(above_threshold)  # has shape (N, 3)
    num_above_threshold = above_threshold_ind.shape[0]
    if num_above_threshold == 0:
        print('WARNING: no voxels above threshold. Second stage will not be used.')
        stage_1_seg_highres = F.interpolate(stage_1_seg[None, None, ...], (D, H, W), align_corners=True, mode='trilinear')[0]
        return stage_1_seg_highres
    elif num_above_threshold > config['inference_max_patches']:
        print(f'WARNING: too many voxels above threshold ({num_above_threshold}). Only {config["inference_max_patches"]} will be used.')
    else:
        print(f'{num_above_threshold} voxels above threshold.')
    print('Running stage 2...')
    t0 = time.time()
    patch_size = config['patch_size']
    padding = patch_size // 2
    x_highres_padded = F.pad(x_highres.to(device), (padding, padding, padding, padding, padding, padding))  # has shape (c, D', H', W')
    patches_x = []
    selections = []
    for j in range(min(num_above_threshold, config['inference_max_patches'])):
        loc_lowres = above_threshold_ind[j].float()
        loc_highres = loc_lowres / torch.tensor([d, h, w], device=device) * torch.tensor([D, H, W], device=device)
        bound_min = (loc_highres - patch_size / 2.).round().int() + padding
        bound_max = bound_min + patch_size
        selection = (slice(None, None, None),)
        selection = selection + tuple([slice(bound_min[i].item(), bound_max[i].item(), None) for i in range(3)])
        patches_x.append(x_highres_padded[selection])
        selections.append(selection)
    patches_x = torch.stack(patches_x, 0)
    loader = torch.utils.data.DataLoader(patches_x, batch_size=4, shuffle=False, drop_last=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            predictions.append(stage_2(batch))
    predictions = torch.cat(predictions, 0).sigmoid()
    if soft_overlap:
        mask_1d = torch.linspace(0, 2 * np.pi, patch_size, device=predictions.device)
        mask_1d = 1 - mask_1d.cos()
        a, b, c = torch.meshgrid(mask_1d, mask_1d, mask_1d, indexing="ij")
        mask = a * b * c
    else:
        mask = torch.ones(patch_size, patch_size, patch_size, device=predictions.device)
    y_pred = torch.zeros(*x_highres_padded.shape, device=predictions.device)
    y_norm = torch.zeros(*x_highres_padded.shape, device=predictions.device)
    for s, patch in zip(selections, predictions):
        y_pred[s] += patch * mask
        y_norm[s] += mask
    output = y_pred / y_norm
    output[torch.isnan(output)] = 0
    output = output[:, padding:-padding, padding:-padding, padding:-padding]
    print(f'Done in {(time.time()-t0):.4f} seconds.')

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='(Required) Path to the Flair')
    parser.add_argument('-o', '--output', help='(Optional) Path to the output')
    parser.add_argument('-m', '--model', help='(Required) Path to the model')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--cpu", help="Use CPU for inference", action="store_true")
    group.add_argument("-g", "--gpu", help="Use GPU for inference", action="store_true")
    args = parser.parse_args()
    t00 = time.time()
    model = args.model
    if not os.path.isfile(model):
        print('ERROR: "model.pth" not found.')
        sys.exit()
    print("model used for inference:", model)
    if not os.path.isfile(args.input):
        print("Error: Input path given does not exist")
        print(args.input)
        sys.exit(1)
    print("input image:", args.input)
    if args.cpu:
        device = 'cpu'
    elif args.gpu:
        device = 'cuda'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model, map_location=device)
    config = checkpoint['config']
    output_file = None
    if args.output:
        if os.path.isfile(args.output):
            output_file = args.output
        elif os.path.isdir(args.output):
            output_file = os.path.join(args.output, 'plexus_mask_' + os.path.basename(args.input))
        else:
            output_file = args.output
    else:
        output_file = os.path.join(os.path.dirname(args.input), 'plexus_mask_' + os.path.basename(args.input))
    print(f'Output segmentation will be saved to "{output_file}".')
    print(f'Opening {args.input}...')
    t0 = time.time()
    t1 = torchio.ScalarImage(args.input)
    print(f'Done in {(time.time()-t0):.4f} seconds.')
    print('Preprocessing...')
    t0 = time.time()
    t1_preprocessed = torchio.transforms.ToCanonical()(t1)
    t1_preprocessed = torchio.transforms.Resample(target=1, image_interpolation='linear')(t1_preprocessed)
    save_shape = t1_preprocessed.shape[1:]
    t1_preprocessed = torchio.transforms.RescaleIntensity(out_min_max=(-1, 1), percentiles=(0.5, 99.5))(t1_preprocessed)

    print(f'Done in {(time.time()-t0):.4f} seconds.')

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

    stage_1.load_state_dict(checkpoint['stage_1_state_dict'])
    stage_2.load_state_dict(checkpoint['stage_2_state_dict'])
    plexus_mask = inference(t1_preprocessed.tensor.float(), config, stage_1, stage_2, soft_overlap=True)
    plexus_mask = plexus_mask.cpu()
    print('Postprocessing...')
    t0 = time.time()
    plexus_mask = torchio.ScalarImage(tensor=plexus_mask, affine=t1_preprocessed.affine)
    plexus_mask_postprocessed = torchio.Resample(args.input)(plexus_mask)
    print(f'Done in {(time.time()-t0):.4f} seconds.')
    plexus_mask_postprocessed["data"] = (plexus_mask_postprocessed["data"] > 0.5).to(torch.uint8)
    plexus_mask_postprocessed.save(output_file)
    print(f'Plexus choroid segmentation was saved to "{output_file}".')
    print(f'Total execution time: {(time.time()-t00):.4f} seconds.')
