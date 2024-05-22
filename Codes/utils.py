#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:12:14 2022

@author: arya.yazdan-panah
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def get_conv_block(dim, in_channels, out_channels, norm, non_lin, kernel_size=3):
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
            m = nn.Sequential(
                get_conv_block(dim=dim, in_channels=block_1_in_channels, out_channels=block_1_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin),
                get_conv_block(dim=dim, in_channels=block_2_in_channels, out_channels=block_2_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin)
            )
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
            m = nn.Sequential(
                get_conv_block(dim=dim, in_channels=block_in_channels, out_channels=block_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin),
                get_conv_block(dim=dim, in_channels=block_out_channels, out_channels=block_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin)
            )
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


def conv_block(
    ichannels,
    ochannels,
    kernel_size=3,
    stride=1,
    padding=1,
    dilatation=1,
    padding_mode="zeros",
):
    block = nn.Sequential(
        nn.Conv3d(
            ichannels,
            ochannels,
            kernel_size,
            stride,
            padding,
            padding_mode=padding_mode,
        ),
        nn.BatchNorm3d(ochannels),
        nn.LeakyReLU(),
    )
    return block


class unet_encoder(nn.Module):
    def __init__(self, stages, n_filtres):
        super(unet_encoder, self).__init__()
        self.module_list = nn.ModuleList()
        for i in range(stages):
            iconv1 = 1 if i == 0 else (2 ** i) * n_filtres
            oconv1 = (2 ** i) * n_filtres
            iconv2 = oconv1
            oconv2 = (2 ** (i + 1)) * n_filtres
            m = nn.Sequential(conv_block(iconv1, oconv1), conv_block(iconv2, oconv2))
            self.module_list.append(m)
        self.pooling = nn.AvgPool3d(2, 2)

    def forward(self, x):
        skip_connections = []
        for i, m in enumerate(self.module_list[:-1]):
            x = m(x)
            skip_connections.append(x)
            x = self.pooling(x)
        x = self.module_list[-1](x)
        return x, skip_connections


class unet_decoder(nn.Module):
    def __init__(self, stages, n_filtres):
        super(unet_decoder, self).__init__()
        self.n_filtres = n_filtres
        self.module_list = nn.ModuleList()
        self.ct = nn.ModuleList()
        for i in reversed(range(stages - 1)):
            iconv1 = (2 ** (i + 1) + (2 ** (i + 2))) * n_filtres
            oconv1 = (2 ** (i + 1)) * n_filtres
            ict = (2 ** (i + 2)) * n_filtres
            self.ct.append(nn.ConvTranspose3d(ict, ict, kernel_size=2, stride=2))
            self.module_list.append(
                nn.Sequential(conv_block(iconv1, oconv1), conv_block(oconv1, oconv1))
            )
        self.final_convdown = nn.Conv3d(2 * n_filtres, 1, 1)

    def forward(self, x, skip_connections):
        for i, (m, c, y) in enumerate(
            zip(self.module_list, self.ct, reversed(skip_connections))
        ):
            x = c(x)
            x = m(torch.cat([y, x], dim=1))
        x = self.final_convdown(x)
        return x


class unet(nn.Module):
    def __init__(self, stages, n_filtres):
        super(unet, self).__init__()
        self.encoder = unet_encoder(stages, n_filtres)
        self.decoder = unet_decoder(stages, n_filtres)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        x = x.sigmoid()
        return x
