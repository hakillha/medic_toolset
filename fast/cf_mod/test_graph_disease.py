#!/usr/bin/env python
# coding: utf-8

# In[1]:

def get_abspath(path):
    path_ = os.path.expanduser(path)
    return os.path.abspath(path_)

# %load /rdfs/fast/home/chenfeng/NetWork/misc/config_init.py
import os, glob, sys, shutil, re, math
# path1 = os.path.expanduser("~/NetWork/torch/segmenter/")
# path1 = os.path.abspath(path1)
# sys.path.insert(0, path1)

# path1 = os.path.expanduser("~/NetWork/tensorflow/classifier/")
# path1 = os.path.abspath(path1)
# sys.path.insert(0, path1)

# path1 = os.path.expanduser("~/NetWork/keras/segmenter/hrnet")
# path1 = os.path.abspath(path1)
# sys.path.insert(0, path1)

# path1 = os.path.expanduser("~/NetWork/misc")
# path1 = os.path.abspath(path1)
# sys.path.insert(0, path1)

for tmp_path in ["segmenter", "misc"]:
    sys.path.insert(0, os.path.join("/rdfs/fast/home/sunyingge/code/chenfeng", tmp_path))

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

import SimpleITK as sitk
import pydicom

import sklearn 
import scipy 
import skimage 

import torch
import keras
import tensorflow as tf
from tensorflow.python.framework import graph_util
from keras import backend as K
K.set_image_data_format("channels_last")

import segmentation_models as sm
# import albumentations as albmt
# from albumentations import (
#     HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
#     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
#     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
#     IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
# )

# def strong_aug(p=0.7):
#     return Compose([
#         OneOf([
# #             CLAHE(clip_limit=2, p=0.2),
#             RandomContrast(limit=0.3, p=0.2),
#             #RandomBrightness(limit=0.3, p=0.3),
#             IAASharpen(alpha=(0.1, 0.4), p=0.3),
#             IAAEmboss(alpha=(0.1, 0.3), strength=(0.1, 0.4), p=0.3),
#         ], p=0.5),
#         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, 
#                          border_mode=cv2.BORDER_CONSTANT, value = -1200,
#                          p=0.2),
#         OneOf([
#             MotionBlur(p=.2),
#             MedianBlur(blur_limit=3, p=.3),
#             Blur(blur_limit=3, p=.2),
#         ], p=0.3),
        
#         OneOf([
#             OpticalDistortion(p=0.3),
#             GridDistortion(p=.2),
#             IAAPiecewiseAffine(p=0.3),
#         ], p=0.5),
#         OneOf([
#             IAAAdditiveGaussianNoise(p=0.5),
#             GaussNoise(p=0.5),
#         ], p=0.4),
#         #HueSaturationValue(p=0.3),
#     ], p=p)
# augmentor = strong_aug()


from utils import *
from my_metrics import *
from functions import *
from config_graphdef import TENSOR_NAME, CONFIG


# In[3]:


CONFIG["pneu1ch_default"] = get_abspath("~/Production/GraphDef/LungPneumonia/HRNET_KERAS_1CH/pneu1ch_default/model.graphdef")
TENSOR_NAME["pneu1ch_default"] = TENSOR_NAME["pneu_default"]

CONFIG["refine039"] = get_abspath("~/Production/GraphDef/LungPneumonia/HRNET_KERAS_1CH/refine039/model.graphdef")
TENSOR_NAME["refine039"] = TENSOR_NAME["pneu_default"]

CONFIG["restart018"] = get_abspath("~/Production/GraphDef/LungPneumonia/HRNET_KERAS_1CH/restart018/model.graphdef")
TENSOR_NAME["restart018"] = TENSOR_NAME["pneu_default"]

CONFIG["reverse_default"] = get_abspath("~/Production/GraphDef/LungPneumonia/HRNET_KERAS_1CH/reverse_default/model.graphdef")
TENSOR_NAME["reverse_default"] = TENSOR_NAME["pneu_default"]

CONFIG["reeval049"] = get_abspath("~/Production/GraphDef/LungPneumonia/HRNET_KERAS_1CH/reeval049/model.graphdef")
TENSOR_NAME["reeval049"] = TENSOR_NAME["pneu_default"]

CONFIG["reeval063"] = get_abspath("~/Production/GraphDef/LungPneumonia/HRNET_KERAS_1CH/reeval063/model.graphdef")
TENSOR_NAME["reeval063"] = TENSOR_NAME["pneu_default"]

CONFIG["default007"] = get_abspath("~/Production/GraphDef/LungPneumonia/HRNET_KERAS_1CH/default007/model.graphdef")
TENSOR_NAME["default007"] = TENSOR_NAME["pneu_default"]

CONFIG["default007"] = get_abspath("/rdfs/fast/home/sunyingge/data/models/work_dir_0512/HRNET_KERAS_DEFAULT_0_00050_20200512-19/model.graphdef")
TENSOR_NAME["default007"] = TENSOR_NAME["pneu_default"]

CONFIG["default007"] = get_abspath("/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/inf/model.graphdef")
# TENSOR_NAME["default007"] = {"SEResUNet/Input/Const":"SEResUNet/Input/Const:0", "SEResUNet/conv_final/BiasAdd":"SEResUNet/conv_final/BiasAdd:0"}
TENSOR_NAME["default007"] = {"inputs":"SEResUNet/Input/Const:0", "outputs":"SEResUNet/conv_final/BiasAdd:0"}

# In[5]:


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95

# This in combination with CONFIG chooses what inf file to use
use_net = "default007"
[graph_dis, sess_dis], [inputs_dis, outputs_dis] = graph_from_graphdef(use_net, config, 
                                                                       CONFIG=CONFIG, TENSOR_NAME=TENSOR_NAME) 


# In[6]:


# info_paths = get_infos("/rdfs/data/chenfeng/Lung/LungPneumonia/patients/TrainSet/batch_0505/normal_pneu_datasets/level4")  + \
#              get_infos("/rdfs/data/chenfeng/Lung/LungPneumonia/patients/TrainSet/batch_0505/normal_pneu_datasets/level4")
info_paths = get_infos("/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/normal_pneu_datasets") + \
    get_infos("/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/covid_pneu_datasets")
# info_paths = get_infos("/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/covid_pneu_datasets")
# info_paths = get_infos("/rdfs/fast/home/sunyingge/data/COV_19/0508/TestSet/normal_pneu_datasets")


# In[7]:


info_paths = sorted(info_paths, key=lambda info:info[0])
print(info_paths[0][0])


# In[8]:


all_result = []
do_paths = info_paths 
need_plot = True
pbar = tqdm(total=len(do_paths))
for num, info in enumerate(do_paths):
    img_file, lab_file = info[0:2]
    try:
        img_ori,  lab_ori  = sitk.ReadImage(img_file, sitk.sitkFloat32), sitk.ReadImage(lab_file, sitk.sitkInt16)
        img_arr,  lab_arr  = sitk.GetArrayFromImage(img_ori), sitk.GetArrayFromImage(lab_ori)
        lab_arr  = np.asarray(lab_arr > 0, dtype=np.uint8)
    except:
        continue
    #lab_arr = pre_lab_arr(lab_arr, description=lab_file, mode="disease").squeeze()
#     lab_arr = lab_arr.astype(np.bool).astype(np.uint8)
    depth, ori_shape = img_arr.shape[0], img_arr.shape[1:]
    spacing = img_ori.GetSpacing()
    min_size = int(24/(np.prod(img_ori.GetSpacing())))
    
    dis_min_ch1 = -1400. ; dis_max_ch1 = 800. 
    dis_min_ch2 = -1000. ; dis_max_ch2 = -100.
    
#     dis_arr = pre_img_arr_2ch(img_arr,
#                              min_ch1=dis_min_ch1, max_ch1=dis_max_ch1,
#                              min_ch2=dis_min_ch2, max_ch2=dis_max_ch2,
#                              )

    dis_arr = pre_img_arr_1ch(img_arr,
                            min_ch1=dis_min_ch1, max_ch1=dis_max_ch1,
                            )
    
    dis_arr = resize3d(dis_arr, (256, 256), interpolation=cv2.INTER_LINEAR)
    dis_arr = np.expand_dims(dis_arr, axis=-1)
    
    pred_ = []
    segs =  64
    assert isinstance(segs, int) and (segs>0) & (segs<70), "Please" 
    step = depth//segs + 1 if depth%segs != 0 else depth//segs
    for ii in range(step):
        if ii != step-1:
            if False: print("stage1")
            if False: print(dis_arr[ii*segs:(ii+1)*segs, ...].shape)
            pp = sess_dis.run(outputs_dis, feed_dict={inputs_dis: dis_arr[ii*segs:(ii+1)*segs, ...]}) #[0]
            pp = 1/ (1 + np.exp(-pp))
            pred_.append(pp)
        else:
            if False: print("stage2")
            if False: print(dis_arr[ii*segs:, ...].shape)
            pp = sess_dis.run(outputs_dis, feed_dict={inputs_dis: dis_arr[ii*segs:, ...]}) #[0]
            pp = 1/ (1 + np.exp(-pp))
            pred_.append(pp)
    dis_prd = np.concatenate(pred_, axis=0)
    dis_prd = (dis_prd > 0.5)
    dis_prd = resize3d(dis_prd.astype(np.uint8), ori_shape, interpolation=cv2.INTER_NEAREST)
    score = dice_coef_pat(dis_prd, lab_arr)
    if score < 0.3:
        print(os.path.dirname(lab_file))
        print(score)
    all_result.append([img_file, score, round(spacing[-1], 1)])
    if False:
        pos = (lab_arr>0.5).any(axis=(-1, -2))
        pos_img = img_arr[pos]; pos_prd = dis_prd[pos]; pos_lab = lab_arr[pos]
        pos_depth = pos_img.shape[0]
        max_depth = min(pos_depth, 20)
        for ii in range(0, max_depth, 2):
            plt.subplot(131)
            plt.imshow(pos_img[ii, ...], cmap="gray")
            plt.subplot(132)
            plt.imshow(pos_img[ii, ...], cmap="gray")
            plt.imshow(pos_prd[ii, ...], alpha=0.5)
            plt.subplot(133)
            plt.imshow(pos_img[ii, ...], cmap="gray")
            plt.imshow(pos_lab[ii, ...], alpha=0.5)
            plt.show()

    if False:
        ori_dir = os.path.dirname(img_file)
        dir_path = ori_dir.replace("", "")
        #assert dir_path is not ori_dir, "check"
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        basename = os.path.basename(img_file)
        pred_dis_ori = sitk.GetImageFromArray(dis_prd)
        pred_dis_ori.CopyInformation(img_ori)
        #shutil.copyfile(img_file, os.path.join(dir_path, basename))
        #sitk.WriteImage(img_ori, os.path.join(dir_path, basename))
        save_file = os.path.join(dir_path, basename.replace(".nii.gz", "_prd-label.nii.gz"))
        assert save_file != img_file
        sitk.WriteImage(pred_dis_ori, save_file)
    
    pbar.update(1)
pbar.close()


# In[9]:


# pickle.dump(all_result, 
#         #    open("/rdfs/fast/home/sunyingge/data/models/work_dir_0512/HRNET_KERAS_DEFAULT_0_00050_20200512-19/train_result_dice_reeval_covid.pkl", "bw")
#     open("/rdfs/fast/home/sunyingge/data/models/work_dir_0512/HRNET_KERAS_DEFAULT_0_00050_20200512-19/train_result_dice_reeval_normal.pkl", "bw"))

print(np.mean([res[1] for res in all_result]))




"""
all_result = [idx for idx in all_result_bak if "covid_pneu_datasets" not in idx[0]]
level0_list = [idx[1] for idx in all_result if( "level0"  in idx[0])]
level1_list = [idx[1] for idx in all_result if( "level1"  in idx[0])]
level2_list = [idx[1] for idx in all_result if( "level2"  in idx[0])]
level3_list = [idx[1] for idx in all_result if( "level3"  in idx[0])]
level4_list = [idx[1] for idx in all_result if( "level4"  in idx[0])]

print(len(level0_list), np.mean(level0_list))
print(len(level1_list), np.mean(level1_list))
print(len(level2_list), np.mean(level2_list))
print(len(level3_list), np.mean(level3_list))
print(len(level4_list), np.mean(level4_list))


# In[15]:


print(np.mean(level0_list + level1_list + level2_list + level3_list + level4_list))


# In[9]:


score_list = [idx[1] for idx in all_result]
print(np.mean(score_list))
print(np.std(score_list))


# In[ ]:


high_count, low_count = 0, 0
high_list, low_list = [], []
high_score_list, low_score_list = [], []
for idx in all_result:
    file, score, spc = idx
    if score < 0.2:
        low_count += 1
        low_list.append(os.path.dirname(file))
        low_score_list.append(score)
    elif score > 0.5:
        high_count += 1
        high_list.append(os.path.dirname(file))
        high_score_list.append(score)
print(high_count)
print(low_count)
        


# In[ ]:


for idx in all_result:
    file, score, spc = idx
    if score > 0.6:
        print(os.path.dirname(file))


# In[ ]:
"""



