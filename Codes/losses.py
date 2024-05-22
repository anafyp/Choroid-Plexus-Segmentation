#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:39:31 2022

@author: arya.yazdan-panah
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.metrics import compute_hausdorff_distance
from pykeops.torch import LazyTensor
from scipy.ndimage.morphology import distance_transform_edt


def erode3D(mask):
    return -F.max_pool3d(-mask, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def dilate3D(mask):
    return F.max_pool3d(mask, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def erode2D(mask):
    return -F.max_pool2d(-mask, (3, 3), (1, 1), (1, 1))


def dilate2D(mask):
    return F.max_pool2d(mask, (3, 3), (1, 1), (1, 1))


def dilate(mask):
    if mask.dim() == 5:
        return dilate3D(mask)
    elif mask.dim() == 4:
        return dilate2D(mask)
    else:
        raise ValueError("Tensor must be 2D or 3D")


def erode(mask):
    if mask.dim() == 5:
        return erode3D(mask)
    elif mask.dim() == 4:
        return erode2D(mask)
    else:
        raise ValueError("Tensor must be 2D or 3D")


def get_edges(mask):
    if mask.dim() == 5:
        ero = erode3D(mask)
    elif mask.dim() == 4:
        ero = erode2D(mask)
    else:
        raise ValueError("Tensor must be 2D or 3D")
    contour = torch.logical_xor(ero > 0.5, mask > 0.5)
    return contour


def Average_Hausdorff_Distance(gth, pred, method="scipy"):
    B = gth.shape[0]
    C = gth.shape[1]
    gth = (gth > 0.5).float()
    pred = (pred > 0.5).float()
    gth_edges = get_edges(gth)
    pred_edges = get_edges(pred)
    hds_losses = []
    for b in range(B):
        for c in range(C):
            gth_edges_bc = gth_edges[b, c, ...]
            pred_edges_bc = pred_edges[b, c, ...]
            if (not gth_edges_bc.sum()) and (not pred_edges_bc.sum()):
                gth2pred = torch.tensor(float('nan'))
                pred2gth = torch.tensor(float('nan'))
                ahd = (gth2pred + pred2gth) / 2
            elif method == "scipy":
                gth_edges_bc = gth_edges_bc.detach().cpu().numpy()
                pred_edges_bc = pred_edges_bc.detach().cpu().numpy()
                gth_edges_bc_dm = distance_transform_edt(1 - gth_edges_bc)
                pred_edges_bc_dm = distance_transform_edt(1 - pred_edges_bc)
                gth2pred = pred_edges_bc_dm[gth_edges_bc].mean()
                pred2gth = gth_edges_bc_dm[pred_edges_bc].mean()
                ahd = (gth2pred + pred2gth) / 2
            elif method == "keops":
                gth_edges_bc_coordinates = torch.nonzero(gth_edges_bc)
                pred_edges_bc_coordinates = torch.nonzero(pred_edges_bc)
                X = LazyTensor(gth_edges_bc_coordinates.view(gth_edges_bc_coordinates.shape[0], 1, gth_edges_bc_coordinates.shape[1]).float())
                Y = LazyTensor(pred_edges_bc_coordinates.view(1, pred_edges_bc_coordinates.shape[0], pred_edges_bc_coordinates.shape[1]).float())
                Distance_matrix = ((X - Y)**2).sum(dim=2)**0.5
                gth2pred = Distance_matrix.min_reduction(1).mean()
                pred2gth = Distance_matrix.min_reduction(0).mean()
                ahd = (gth2pred + pred2gth) / 2
                ahd = ahd.cpu().numpy()
            hds_losses.append(ahd)
    return np.array(hds_losses).mean()


class AverageHausdorffLoss(nn.Module):
    def __init__(self, device='cpu', method="scipy", ignore_nan=True):
        super(AverageHausdorffLoss, self).__init__()
        self.device = device
        self.method = method
        self.ignore_nan = ignore_nan

    def forward(self, gth, pred):
        B = gth.shape[0]
        C = gth.shape[1]
        gth = (gth > 0.5).float()
        pred = (pred > 0.5).float()
        gth_edges = get_edges(gth)
        pred_edges = get_edges(pred)
        hds_losses = []
        for b in range(B):
            for c in range(C):
                gth_edges_bc = gth_edges[b, c, ...]
                pred_edges_bc = pred_edges[b, c, ...]
                if (not gth_edges_bc.sum()) and (not pred_edges_bc.sum()):
                    gth2pred = torch.tensor(float('nan'))
                    pred2gth = torch.tensor(float('nan'))
                elif self.method == "scipy":
                    gth_edges_bc = gth_edges_bc.detach().cpu().numpy()
                    pred_edges_bc = pred_edges_bc.detach().cpu().numpy()
                    gth_edges_bc_dm = distance_transform_edt(1 - gth_edges_bc)
                    pred_edges_bc_dm = distance_transform_edt(1 - pred_edges_bc)
                    gth2pred = pred_edges_bc_dm[gth_edges_bc].mean()
                    pred2gth = gth_edges_bc_dm[pred_edges_bc].mean()
                elif self.method == "keops":
                    gth_edges_bc_coordinates = torch.nonzero(gth_edges_bc)
                    pred_edges_bc_coordinates = torch.nonzero(pred_edges_bc)
                    X = LazyTensor(gth_edges_bc_coordinates.view(gth_edges_bc_coordinates.shape[0], 1, gth_edges_bc_coordinates.shape[1]).float())
                    Y = LazyTensor(pred_edges_bc_coordinates.view(1, pred_edges_bc_coordinates.shape[0], pred_edges_bc_coordinates.shape[1]).float())
                    Distance_matrix = ((X - Y)**2).sum(dim=2)**0.5
                    gth2pred = Distance_matrix.min_reduction(1).mean()
                    pred2gth = Distance_matrix.min_reduction(0).mean()
                ahd = (gth2pred + pred2gth) / 2
                hds_losses.append(torch.tensor(1 - (1 / (1 + ahd))))
        if self.ignore_nan:
            return torch.nanmean(torch.stack(hds_losses))
        return torch.mean(torch.stack(hds_losses))


def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    if len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    else:
        raise ValueError("Can only process 3D images")


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def hard_open(img):
    return dilate(erode(img))


def soft_skel(img, max_iter=100):
    skel = F.relu(img - soft_open(img))
    iteration = 0
    with torch.no_grad():
        while True and iteration < max_iter:
            iteration += 1
            img = soft_erode(img)
            # img = erode(img)
            delta = F.relu(img - soft_open(img))
            # delta = F.relu(img - hard_open(img))
            to_add = F.relu(delta - skel * delta)
            if not to_add.sum():
                break
            skel = skel + F.relu(delta - skel * delta)
    return skel


class soft_cldice(nn.Module):
    def __init__(self, iter_=100, smooth=10e-7):
        super(soft_cldice, self).__init__()
        self.iter_ = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter_)
        skel_true = soft_skel(y_true, self.iter_)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


class Skel_Hausdorff_Loss(nn.Module):
    def __init__(self, alpha=0.5, iter_=100, smooth=10e-7, device='cpu'):
        super(Skel_Hausdorff_Loss, self).__init__()
        self.alpha = alpha
        self.skel_loss = soft_cldice(iter_, smooth)
        self.ahd_loss = AverageHausdorffLoss(device=device)
        self.__name__ = "Skel_Hausdorff_Loss"

    def forward(self, y_true, y_pred):
        cl_dice = self.skel_loss(y_true, y_pred)
        hd = self.ahd_loss(y_true, y_pred)
        return cl_dice, hd, ((1 - self.alpha) * cl_dice) + (self.alpha * hd)


class HausdorffLoss(nn.Module):
    def __init__(self, batch=True):
        super(HausdorffLoss, self).__init__()
        self.batch = batch

    def forward(self, inputs, targets):
        inputs = inputs > 0.5
        targets = targets > 0.5
        # Softmax ?
        if not self.batch:
            inputs, targets = inputs.unsqueeze(0), targets.unsqueeze(0)
        hds = compute_hausdorff_distance(inputs, targets)
        to_mean = torch.zeros(hds.shape)
        for i in range(len(hds)):
            if torch.isnan(hds[i]):
                to_mean[i] = 1
            else:
                to_mean[i] = 1 - (1 / (1 + hds[i]))
        loss = torch.mean(to_mean)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class ContinuousDiceLoss(nn.Module):
    def __init__(self, smooth=0, reduction="mean", batch=True):
        super(ContinuousDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.batch = batch

    def forward(self, inputs, targets):
        sdice = 0
        if not self.batch:
            inputs, targets = inputs.unsqueeze(0), targets.unsqueeze(0)
        for idx in range(inputs.shape[0]):
            intersection = torch.minimum(inputs[idx, ...], targets[idx, ...]).sum()
            dice = (2.0 * intersection + self.smooth) / (
                inputs[idx, ...].sum() + targets[idx, ...].sum() + self.smooth
            )
            sdice += dice
        if self.reduction == "mean":
            sdice = sdice / inputs.shape[0]
        return 1 - sdice


class ver(nn.Module):
    def __init__(self, reduction="mean"):
        super(ver, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)
        maver = 0
        for idx in range(inputs.shape[0]):
            v1 = torch.sum(inputs[idx, ...])
            v2 = torch.sum(targets[idx, ...])
            maver += (v2 - v1) / v2
        if self.reduction == "mean":
            maver = maver / inputs.shape[0]
        return maver


class aver(nn.Module):
    def __init__(self, reduction="mean"):
        super(aver, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)
        maver = 0
        for idx in range(inputs.shape[0]):
            v1 = torch.sum(inputs[idx, ...])
            v2 = torch.sum(targets[idx, ...])
            maver += torch.abs((v2 - v1) / v2)
        if self.reduction == "mean":
            maver = maver / inputs.shape[0]
        if maver > 1:
            maver = maver / maver
        return maver


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss
        return Dice_BCE


class ContinuousDiceBCELoss(nn.Module):
    def __init__(self, smooth=0):
        super(ContinuousDiceBCELoss, self).__init__()
        self.smooth = smooth
        self.__name__ = "ContinuousDiceBCE"

    def forward(self, x, y, full_output=False):
        BCELoss = nn.BCEWithLogitsLoss(reduction="mean")(x, y)
        x = torch.sigmoid(x)
        DICELoss = ContinuousDiceLoss(self.smooth, reduction="mean")(x, y)
        if full_output:
            return DICELoss + BCELoss, BCELoss, DICELoss
        return DICELoss + BCELoss


def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [2, 3, 4]
    # Two dimensional
    elif len(shape) == 4:
        return [2, 3]
    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


class SymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.6, gamma=0.5, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        # axis = [2, 3, 4]
        axis = identify_axis(y_pred.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        TI = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
        # Calculate losses separately for each class, enhancing both classes
        loss = torch.pow(1 - TI[:], self.gamma)
        # Average class scores
        loss = torch.mean(torch.stack([loss], axis=-1))
        return loss


class SymmetricFocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.6, gamma=.5, epsilon=1e-07):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        # Calculate losses separately for each class
        ce = self.delta * torch.pow(1 - y_pred, 1 - self.gamma) * cross_entropy
        loss = torch.mean(torch.sum(torch.stack([ce], axis=-1), axis=-1))
        return loss


class SymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.__name__ = "SymmetricUnifiedFocalLoss"

    def forward(self, y_pred, y_true):
        symmetric_ftl = SymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)
        symmetric_fl = SymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)
        loss = (self.weight * symmetric_ftl) + ((1 - self.weight) * symmetric_fl)
        return symmetric_ftl, symmetric_fl, loss


class ComboLoss(nn.Module):
    def __init__(self, smooth=0):
        super(ComboLoss, self).__init__()
        self.smooth = smooth
        self.__name__ = "ComboLoss"

    def forward(self, x, y, full_output=False):
        BCELoss = nn.BCELoss(reduction="mean")(x, y)
        DICELoss = ContinuousDiceLoss(self.smooth, reduction="mean")(x, y)
        return DICELoss, BCELoss, (DICELoss + BCELoss) / 2


class GeomComboContinuous(nn.Module):
    def __init__(self, smooth=10e-7, alpha=0.25, beta=0.25, gamma=0.25, delta=0.25):
        super(GeomComboContinuous, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.__name__ = "GeomComboContinuous"

    def forward(self, x, y, y_prime):
        BCELoss = nn.BCELoss(reduction="mean")(x, y)
        DICELoss = ContinuousDiceLoss(self.smooth, reduction="mean")(x, y)
        HDLoss = HausdorffLoss()(x, y)
        Closs = ContinuousDiceLoss(self.smooth, reduction="mean")(y_prime, y)
        loss = ((self.alpha * BCELoss) + (self.beta * DICELoss) + (self.gamma * HDLoss) + (self.delta * Closs)) / 4
        return BCELoss, DICELoss, HDLoss, Closs, loss


class DiceCLDice(nn.Module):
    def __init__(self, smooth=10e-7, alpha=0.5):
        super(DiceCLDice, self).__init__()
        self.dice = ContinuousDiceLoss()
        self.cldice = soft_cldice()
        self.alpha = alpha
        self.__name__ = "DiceCLDice"

    def forward(self, x, y):
        DICELoss = self.dice(x, y)
        CLDICEloss = self.cldice(x, y)
        return DICELoss, CLDICEloss, ((1 - self.alpha) * CLDICEloss) + (self.alpha * DICELoss)


class DiceCLDiceAHD(nn.Module):
    def __init__(self, smooth=10e-7, alpha=0.5):
        super(DiceCLDiceAHD, self).__init__()
        self.dice = ContinuousDiceLoss()
        self.cldice = soft_cldice()
        self.ahd = AverageHausdorffLoss()
        self.alpha = alpha
        self.__name__ = "DiceCLDiceAHD"

    def forward(self, x, y):
        DICELoss = self.dice(x, y)
        CLDICEloss = self.cldice(x, y)
        AHDloss = self.ahd(x, y)
        return DICELoss, CLDICEloss, AHDloss, (CLDICEloss + DICELoss + AHDloss) / 3
