#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


# %load /rdfs/fast/home/chenfeng/NetWork/misc/config_init.py
import os, glob, sys, shutil, re, math

for tmp_path in ["segmenter", "misc"]:
    sys.path.insert(0, os.path.join("/rdfs/fast/home/sunyingge/code/chenfeng", tmp_path))

from utils import *
from functions import *
from config_graphdef import TENSOR_NAME, CONFIG
from hrnet import HighResolutionNet
from data_tools import paths_for_dataset, BaseDataset, KerasLoader, paths_for_dataset

import logging

from importlib import import_module
import numpy as np
import SimpleITK as sitk
import cv2
from tqdm import tqdm
import subprocess as sp
import multiprocessing as mp
#import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import time, datetime
import csv
import pickle

import sklearn as skn
import scipy as sci
import skimage as skm

import keras
import tensorflow as tf
from tensorflow.python.framework import graph_util
from keras import backend as K
K.set_image_data_format("channels_last")

import segmentation_models as sm


# In[3]:


class Args(object):
    def __init__(self):
        self.gpu = "0"
        self.n_channels = 2
        self.n_classes = 1
        self.epochs = 80
        self.initial_epoch = 0
        self.global_step = 0
        self.batchsize = 64
        self.lr = 0.0005
        self.percentage = 0.1
        self.load = False
        self.save_cp = True
        self.img_size = (256, 256)
        self.description = "HRNET_KERAS_DEFAULT"
args = Args()
localtime = time.strftime("%Y%m%d-%H", time.localtime())
description = "{}_{:.5f}_{}".format(args.description, args.lr, localtime)
print(description)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
model_dir = os.path.join("/rdfs/fast/home/sunyingge/data/models/work_dir_0512", 
                        description,
            )
log_dir = os.path.join("/rdfs/fast/home/sunyingge/data/models/work_dir_0512", 
                        description,
            )
print(model_dir)
print(log_dir)


# In[4]:


flag_set = ['level0', 'level1', 'level2', "level3", "level4", "healthy"]
flag_set = ["datasets"]

print("==>>train: ")
train_all, train_pos, train_neg = paths_for_dataset("/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train/", 
                 flags=["train"],
                 seed=999,
                 isprint=True)

print("==>>test: ")
test_all, test_pos, test_neg   = paths_for_dataset("/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/TestSet/", 
                 flags=["test"],
                 seed=999,
                 isprint=True)

print("==>>val: ")
val_all, val_pos, val_neg     = paths_for_dataset("/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Val/", 
                 flags=["val"],
                 seed=999,
                 isprint=True)

#train_paths_cons  = train_pos["level0"] + train_pos["level1"] + train_pos["level4"] 
#train_paths_chan  = train_pos["level2"] + train_pos["level3"] + train_all["healthy"]

#val_paths_cons    = val_pos["level0"] + val_pos["level1"] + val_pos["level3"] + val_pos["level4"]
#val_paths_chan    = val_pos["level2"] + val_all["healthy"]

#test_paths        = val_pos["level0"] + val_pos["level1"] + val_pos["level2"] +val_pos["level3"] + val_pos["level4"]

train_paths_cons  = train_pos["train"]
train_paths_chan  = []

val_paths_cons    = val_pos["val"]
val_paths_chan    = []

test_paths        = test_pos["test"]


np.random.seed(999)
train_paths_cons = np.random.permutation(train_paths_cons).tolist()
train_paths_chan = np.random.permutation(train_paths_chan).tolist()
val_paths_cons   = np.random.permutation(val_paths_cons).tolist()
val_paths_chan   = np.random.permutation(val_paths_chan).tolist()
test_paths  = np.random.permutation(test_paths).tolist()

print("++"*30)
print(f"Length of train_paths_cons: {len(train_paths_cons)}")
print(f"Length of train_paths_chan: {len(train_paths_chan)}")

print(f"Length of val_paths_cons: {len(val_paths_cons)}")
print(f"Length of val_paths_chan: {len(val_paths_chan)}")

print(f"Length of test_paths: {len(test_paths)}")


def preprocess(**args):
    img, lab = args["image"], args["mask"]
    args["image"] = (255.*img).astype(np.uint8)
    args["mask"]  = lab
    return args
def proprocess(**args):
    img, lab = args["image"], args["mask"]
    args["image"] = (img/255.).astype(np.float32)
    args["mask"]  = lab
    return args 

train_dataset = BaseDataset(train_paths_cons, train_paths_chan, img_size=args.img_size,choice="all",
                          image_key="image", mask_key="mask", 
#                           augmentation=augmentor,
#                           preprocess=preprocess, proprocess=proprocess, 
                       )
val_dataset = BaseDataset(val_paths_cons, val_paths_chan, img_size=args.img_size,choice="all",
                          image_key="image", mask_key="mask", 
#                         augmentation=augmentor,
#                           preprocess=preprocess, proprocess=proprocess,
                       )
test_dataset = BaseDataset(test_paths, img_size=args.img_size,choice="all",
                          image_key="image", mask_key="mask", 
#                         augmentation=augmentor,
#                           preprocess=preprocess, proprocess=proprocess,
                        )
print(f"train_dataset: {len(train_dataset)}")
print(f"val_dataset: {len(val_dataset)}")
print(f"test_dataset: {len(test_dataset)}")


# In[5]:


train_loader = KerasLoader(train_dataset, factor=0.2, batch_size=args.batchsize, shuffle=True)
val_loader = KerasLoader(val_dataset, factor=0.2, batch_size=args.batchsize, shuffle=True)
test_loader = KerasLoader(test_dataset, factor=1.0, batch_size=args.batchsize, shuffle=False)
print(f"train_loader: {len(train_loader)}")
print(f"val_loader: {len(val_loader)}")
print(f"test_loader: {len(test_loader)}")

model = HighResolutionNet(input_size=(256, 256, 1), num_classes=2)

if not os.path.exists(model_dir): os.makedirs(model_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)


# In[6]:


dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss(alpha=0.75, gamma=2.) #if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + focal_loss
optim = keras.optimizers.Adam(args.lr)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(optim, total_loss, metrics)

checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')) 
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1)
logging = keras.callbacks.TensorBoard(log_dir=log_dir)


# In[ ]:


if False:
    model.load_weights("/rdfs/fast/home/chenfeng/Documents/Models/keras/LungPneumonia/HRNET_KERAS_DEFAULT_0.00050_20200506-15/ep007-loss0.172-val_loss0.198.h5")

saved = model.fit_generator(
    train_loader,
    steps_per_epoch=len(train_loader),
    validation_data=val_loader,
    epochs=args.epochs,
    initial_epoch=0,
    callbacks=[checkpoint, reduce_lr, logging]
)

model.save_weights(os.path.join(model_dir, 'final_weights.h5'))
model.load_weights(os.path.join(model_dir, 'final_weights.h5'))
scores = model.evaluate_generator(test_loader)
print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

model.save(os.path.join(model_dir, 'final_weights_with_models.h5'))


# In[ ]:




