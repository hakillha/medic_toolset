#coding:utf-8
"""
author: feng
date:   2020-04-27
"""

import sys, os, re, math, glob
from collections import defaultdict
import numpy as np

import skimage
import scipy

import tensorflow as tf

_EPS = 1.e-6

def dice_coef_sli(y_true, y_pred, axis=(1,2,3)):
    _EPS = 1.e-6
    intersection = np.sum(y_true * y_pred, axis=axis)
    union = np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis)
    dice = (2. * intersection+_EPS) / (union +_EPS)
    dice = np.mean(dice, axis=0)
    return dice

def dice_coef_pat(y_true, y_pred, axis=None):
    _EPS = 1.e-6
    intersection = np.sum(y_true * y_pred, axis=axis)
    union = np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis)
    dice = (2. * intersection+_EPS) / (union +_EPS)
    return dice

def dice_coef_tf(y_true, y_pred, axis=None):
    _EPS = 1.e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
    dice = (2. * intersection+_EPS) / (union + _EPS)
    #dice = tf.reduce_sum(dice, axis=0)
    return dice

class SegPrRe(object):
    def __init__(self,
        #method="precision",
        level="lesion",
        #base="patient",
        threshold=None,
    ):
        level = level.lower()
        assert level in ["lesion", "pixel"], "Please Initial The Object With Avaliable LEVEL"
        self.level = method
        self.threshold=threshold

        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.tn = 0

        self.inter = 0
        self.union = 0

        def _pixel_level(self, pr, gt):
            _FP, _TP, _FN = 0, 0, 0
            if self.threshold:
                pr = np.asarray(pr>self.threshold, dtype=np.uint8)
            _TP = np.sum(pr*gt)
            _FP = np.sum(pr*(1-gt))
            #_TN = np.sum((1-pr)*(1-gt))
            _FN = np.sum((1-pr)*gt)
            return _FP, _TP, _FN


        def _lesion_level(self, pr, gt):
            _FP, _TP, _FN = 0, 0, 0
            if self.threshold:
                pr = np.asarray(pr>self.threshold, dtype=np.uint8)
            prd, prd_num = skimage.measure.label(pr, return_num=True, connectivity=2)
            msk, msk_num = skimage.measure.label(gt, return_num=True, connectivity=2)
            for i in range(1, prd_num+1):
                prd_binary_i = np.asarray(prd==i, dtype=np.uint8)
                msk_match_left = msk * prd_binary_i
                max_value = np.nanmax(msk_match_left)
                if max_value == 0: _FP += 1; continue
                values = set()
                for v in range(1, int(max_value)):
                    if (msk_match_left==v).any(): values.add(v)
                if len(values) == 0:
                    msk_match_binary = np.where(mask==max_value, 1, 0)
                    inter = float(np.sum(msk_match_binary*prd_binary_i))
                    union = float(np.sum(np.where((mask_match_binary + prd_binary_i)>=1, 1, 0))) + _EPS
                    if inter/union >= self.threshold:
                        _TP += 1
                    else:
                        _FP += 1
                else:
                    fp_flag = True
                    for v in values:
                        msk_match_binary = np.where(msk==v, 1, 0)
                        inter = float(np.sum(msk_match_binary * prd_binary_i))
                        union = float(np.sum(np.where((msk_matck_binary + prd_binary_i)>=1, 1, 0))) + _EPS
                        if inter/union >= self.threshold:
                            _TP += 1
                            fp_flag == False
                            break
                    if fp_flag == True: _FP += 1
            _FN = msk_num - _TP
            return _FP, _TP, _FN


        def __call__(self, pr, gt):
            if self.level == "lesion":
                _FP, _TP, _FN = self._lesion_level(pr, gt)
            elif self.level == "pixel":
                _FP, _TP, _FN = self._pixel_level(pr, gt)

            self.fp += _FP
            self.tp += _TP
            self.fn += _FN

            precision = _TP/(_TP + _FP + _EPS)
            recall    = _TP/(_TP + _FN + _EPS)
        
            return precision, recall 

        def add_(self, other):
            self.fp     = self.fp + other.fp
            self.tp     = self.tp + other.tp
            self.fn     = self.fn + other.fn
            self.tn     = self.tn + other.tn
            self.inter  = self.inter + other.inter
            self.union  = self.union + other.union

        def get_fp(self):
            return self.fp
        def get_tp(self):
            return self.tp
        def get_fn(self):
            return self.fn
        def get_tn(self):
            return self.tn
        def get_inter(self):
            return self.inter
        def get_union(self):
            return self.union
        def get_precision_recall(self):
            precision = self.tp/(self.tp + self.fp + _EPS)
            recall    = self.tp/(self.tp + self.fn + _EPS)
            return precision, recall

