#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:13:40 2022

@author: arya.yazdan-panah
"""

import os
import numpy as np
import pandas as pd
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import shutil


root_dir = "......"
dataset_infos = os.path.join(root_dir, "subjects_infos.csv")
all_subjects_dir = os.path.join(root_dir, "data_torch")
test_dir = os.path.join(root_dir, "test_torch")
train_dir = os.path.join(root_dir, "train_torch")
df = pd.read_csv(dataset_infos)

# Stratify at the subject level and not separating multiple visits of the same subject in train/test/val
subjects = df["SUBJECT"].unique()
to_strat = []
for s in subjects:
    to_strat.append(df[(df["SUBJECT"] == s) & (df["VISIT"] == "V01")]["Useful info"].values[0])

train, test = train_test_split(subjects, test_size=0.2, random_state=42, stratify=to_strat)

for t in train:
    tc = glob.glob(os.path.join(all_subjects_dir, t + "*"))
    for c in tc:
        shutil.copy(c, train_dir)

for t in test:
    tc = glob.glob(os.path.join(all_subjects_dir, t + "*"))
    for c in tc:
        shutil.copy(c, test_dir)


subjects = df["SUBJECT"].unique()
to_strat = []
for s in train:
    to_strat.append(df[(df["SUBJECT"] == s) & (df["VISIT"] == "V01")]["Useful info"].values[0])
np.random.seed(42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = []
for train_index, test_index in skf.split(subjects, to_strat):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = subjects[train_index], subjects[test_index]
    np.random.shuffle(X_train)
    np.random.shuffle(X_test)
    folds.append({"train": X_train, "val": X_test})

with open(os.path.join(train_dir, 'folds.pickle'), 'wb') as file:
    pickle.dump(folds, file)
