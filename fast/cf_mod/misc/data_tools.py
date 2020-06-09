#_*_ coding:utf-8 _*_

"""
author: feng
date:   2020-03-27
target: for keras

"""
import os, sys, glob, math, re

from collections import defaultdict
import numpy as np
import cv2
import pickle
import SimpleITK as sitk

import scipy

import tensorflow as tf
import keras

EPS = 1.0e-8

def paths_for_dataset(dir_path, 
                        flags=["covid_pneu", "normal_pneu", "healthy"], 
                        seed=999, 
                        isprint=False):
   
    np.random.seed(seed)
    all_ret_dict = defaultdict(list)
    pos_ret_dict = defaultdict(list)
    neg_ret_dict = defaultdict(list)
    for root, dirs, files in os.walk(dir_path):
        if len(files)>100:
            all_paths = glob.glob(os.path.join(root, "*.pkl"))
            pos_paths = glob.glob(os.path.join(root, "*_pos-*.pkl"))
            neg_paths = glob.glob(os.path.join(root, "*_neg-*.pkl"))
            # print(len(all_paths))
            # print(len(pos_paths))
            # print(len(neg_paths))
            for key in flags:
                if key in root.lower():
                    all_ret_dict[key] += all_paths
                    pos_ret_dict[key] += pos_paths
                    neg_ret_dict[key] += neg_paths
                    break
    for key in flags:
        all_ret_dict[key] = np.random.permutation(all_ret_dict[key]).tolist()
        pos_ret_dict[key] = np.random.permutation(pos_ret_dict[key]).tolist()
        neg_ret_dict[key] = np.random.permutation(neg_ret_dict[key]).tolist()
        if isprint:
            print(">>"*20+f"{key}"+"<<"*20)
            print(f"all: {len(all_ret_dict[key])}") 
            print(f"pos: {len(pos_ret_dict[key])}") 
            print(f"neg: {len(neg_ret_dict[key])}") 
    if isprint: print("====="*20)
    return all_ret_dict, pos_ret_dict, neg_ret_dict

    

def get_patches(img, input_depth, step_size, dtype=np.float32):
    r"""
    img: 4 dim numpy.array, [depth, height, width, channels]
      or 3 dim numpy.array, [depth, height, width]
    return: 5 dim numpy.array, [patch_num, input_depth, height, width, channels]
      or    4 dim numpy.array, [patch_num, input_depth, height, width]
    """
    
    assert img.ndim == 4 or img.ndim == 3, "Please Check the dimension of the image_array"
    if img.ndim == 4:
        patch_num = math.ceil((img.shape[0]-input_depth)/step_size) + 1
        img_patchs = np.zeros([patch_num, input_depth, img.shape[1], img.shape[2], img.shape[3]], dtype=dtype)
        for k in range(patch_num):
            img_tmp = np.zeros([input_depth, img.shape[1], img.shape[2], img.shape[3]], dtype=dtype)
            if k*step_size+input_depth > img.shape[0]:
                img_tmp[...] = img[img.shape[0]-input_depth:img.shape[0], ...]
            else:
                end = int(np.amin((k*step_size+input_depth, img.shape[0])))
                img_tmp[...] = img[k*step_size:end, ...]
            img_patchs[k, ...] = img_tmp
        return img_patchs
    elif img.ndim == 3:
        patch_num = math.ceil((img.shape[0]-input_depth)/step_size) + 1
        img_patchs = np.zeros([patch_num, input_depth, img.shape[1], img.shape[2]], dtype=dtype)
        for k in range(patch_num):
            img_tmp = np.zeros([input_depth, img.shape[1], img.shape[2]], dtype=dtype)
            if k*step_size+input_depth > img.shape[0]:
                img_tmp[...] = img[img.shape[0]-input_depth:img.shape[0], ...]
            else:
                end = int(np.amin((k*step_size+input_depth, img.shape[0])))
                img_tmp[...] = img[k*step_size:end, ...]
            img_patchs[k, ...] = img_tmp
        return img_patchs
    
def merge_patches(patches, ori_arr, step_size=8):
    num = patches.shape[0]
    input_depth = patches.shape[1]
    temp = np.zeros_like(ori_arr)
    for ii in range(num-1):
        temp[step_size*ii:step_size*ii+input_depth] = patches[ii]
    temp[-input_depth:] = patches[-1]
    return temp

## make slice_pkl for 2D segmentation   
def extract_slice_1ch(info, to_dir, sub_dir="new_test", \
                min_ct_1ch=-1200, max_ct_1ch=800,  \
                pro_fn= None,         \
                phase="xiamen", tofor="disease", isprint=False):
    
    assert tofor in ["disease", "lobe"], "tofor should be disease or lobe"
    if tofor == "disease":
        assert phase in ["xiamen", "301", "covid", "others"], "phase should be xiamen, 301, or covid"
    if tofor == "lobe":
        assert phase in ["label",  "others"], "phase should be xiamen, 301, covid, or luna"
    wrong = 0
    to_dir = os.path.join(to_dir, sub_dir) 
    if not os.path.exists(to_dir): os.makedirs(to_dir)
    img_file, lab_file = info[:2]
    name_list = img_file.split("/")
    basename = "_".join([*name_list[-6:-1], name_list[-1][:-7].replace(" ", "").replace("-", ""), "{}.pkl"]).replace(" ", "-")
    try:
        if isprint: print(basename)
        img_ori = sitk.ReadImage(img_file, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img_ori)
        lab_ori = sitk.ReadImage(lab_file, sitk.sitkUInt16)
        lab = sitk.GetArrayFromImage(lab_ori)
        if pro_fn:
            img, lab = pro_fn(img, lab, lab_ori)
        ## normalization
        img_1ch = np.clip(img, min_ct_1ch, max_ct_1ch)
        img_1ch = (img_1ch-np.amin(img_1ch)+EPS)/(np.amax(img_1ch)-np.amin(img_1ch)+EPS)
        
        img = np.expand_dims(img_1ch, axis=-1) # shape:d, h, w, c
        ## ordering the label for disease of various hospital
        if tofor == "disease":
            if phase=="xiamen":
                lab[(lab>71)*(lab<78)] = 71
                lab[lab!=71] = 0
                lab[lab==71] = 1
            elif phase=="301":
                lab[lab!=71] = 0
                lab[lab==71] = 1
            elif phase=="covid":
                lab[(lab>194)*(lab<300)] = 71
                lab[lab!=71] = 0
                lab[lab==71] = 1
            else:
                pos         = (lab>=1)
                lab[pos]    = 1
                lab[~pos]   = 0
        elif tofor == "lobe":
            if phase=="label":
                #lab[(lab<219)+(lab>223)] = 0
                pos = ((lab>0)*(lab<=5)) + ((lab<=224)*(lab>=219))
                lab[~pos] = 0
                lab[lab==219] = 1
                lab[lab==220] = 2
                lab[lab==221] = 3
                lab[lab==222] = 4
                lab[lab==223] = 5
            elif phase=="others":
                pass
        pos_mask, neg_mask = None, None
        # pick out the channels/depths
        if tofor == "disease":
            pos_mask = (lab == 1).any(axis=(-1, -2))
        elif tofor == "lobe":
            pos_mask = (lab > 0).any(axis=(-1, -2))
        neg_mask = ~pos_mask
        if isprint: print(">>>", "/".join(name_list[-4:-1]), ": ")
        pos_img = img[pos_mask]
        pos_lab = lab[pos_mask]
        if isprint: print("positive: ", pos_img.shape, end="  ")
        neg_img = img[neg_mask]
        neg_lab = lab[neg_mask]
        if isprint: print("negative: ", neg_img.shape )
        for zz in range(pos_img.shape[0]):
            sli_img = pos_img[zz, ...]
            sli_lab = pos_lab[zz, ...]
            file_dict = {"image":sli_img, "mask":sli_lab}
            new_name = os.path.join(to_dir, basename.format("pos-"+str(zz)))
            pickle.dump(file_dict, open(new_name, "wb"))
        for zz in range(neg_img.shape[0]):
            sli_img = neg_img[zz, ...]
            sli_lab = neg_lab[zz, ...]
            file_dict = {"image":sli_img, "mask":sli_lab}
            new_name = os.path.join(to_dir, basename.format("neg-"+str(zz)))
            pickle.dump(file_dict, open(new_name, "wb"))
    except Exception as erro:
        wrong += 1
        if isprint: print(erro); print("***"*20)
    if isprint: print("All Wrongs: {}".format(wrong))

def extract_slice_2ch(info, to_dir, sub_dir="new_test", \
                min_ct_1ch=-1200, max_ct_1ch=800,             \
                min_ct_2ch=-1000,  max_ct_2ch=-100,            \
                pro_fn= None,         \
                phase="xiamen", tofor="disease", isprint=False):
    
    assert tofor in ["disease", "lobe"], "tofor should be disease or lobe"
    if tofor == "disease":
        assert phase in ["xiamen", "301", "covid", "others"], "phase should be xiamen, 301, or covid"
    if tofor == "lobe":
        assert phase in ["label", "others"], "phase should be xiamen, 301, covid, or luna"
    wrong = 0
    to_dir = os.path.join(to_dir, sub_dir) 
    if not os.path.exists(to_dir): os.makedirs(to_dir)
    img_file, lab_file = info[:2]
    name_list = img_file.split("/")
    basename = "_".join([*name_list[-6:-1], name_list[-1][:-7].replace(" ", "").replace("-", ""), "{}.pkl"]).replace(" ", "-")
    try:
        if isprint: print(basename)
        img_ori = sitk.ReadImage(img_file, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img_ori)
        lab_ori = sitk.ReadImage(lab_file, sitk.sitkUInt16)
        lab = sitk.GetArrayFromImage(lab_ori)
        if pro_fn:
            img, lab = pro_fn(img, lab, lab_ori)
        ## normalization
        img_1ch = np.clip(img, min_ct_1ch, max_ct_1ch)
        img_1ch = (img_1ch-np.amin(img_1ch)+EPS)/(np.amax(img_1ch)-np.amin(img_1ch)+EPS)
        
        img_2ch = np.clip(img, min_ct_2ch, max_ct_2ch)
        img_2ch = (img_2ch-np.amin(img_2ch)+EPS)/(np.amax(img_2ch)-np.amin(img_2ch)+EPS)
        
        img = np.stack([img_1ch, img_2ch], axis=-1) # shape:d, h, w, c
        
        ## ordering the label for disease of various hospital
        if tofor == "disease":
            if phase=="xiamen":
                lab[(lab>71)*(lab<78)] = 71
                lab[lab!=71] = 0
                lab[lab==71] = 1
            elif phase=="301":
                lab[lab!=71] = 0
                lab[lab==71] = 1
            elif phase=="covid":
                lab[(lab>194)*(lab<300)] = 71
                lab[lab!=71] = 0
                lab[lab==71] = 1
            else:
                pos         = (lab>=1)
                lab[pos]    = 1
                lab[~pos]   = 0
        elif tofor == "lobe":
            if phase=="label":
                #lab[(lab<219)+(lab>223)] = 0
                pos = ((lab>0)*(lab<=5)) + ((lab<=224)*(lab>=219))
                lab[~pos] = 0
                lab[lab==219] = 1
                lab[lab==220] = 2
                lab[lab==221] = 3
                lab[lab==222] = 4
                lab[lab==223] = 5
            elif phase=="others":
                pass
        pos_mask, neg_mask = None, None
        if tofor == "disease":
            pos_mask = (lab == 1).any(axis=(-1, -2))
        elif tofor == "lobe":
            pos_mask = (lab > 0).any(axis=(-1, -2))
        neg_mask = ~pos_mask
        if isprint: print(">>>", "/".join(name_list[-4:-1]), ": ")
        pos_img = img[pos_mask]
        pos_lab = lab[pos_mask]
        if isprint: print("positive: ", pos_img.shape, end="  ")
        neg_img = img[neg_mask]
        neg_lab = lab[neg_mask]
        if isprint: print("negative: ", neg_img.shape )
        for zz in range(pos_img.shape[0]):
            sli_img = pos_img[zz, ...]
            sli_lab = pos_lab[zz, ...]
            file_dict = {"image":sli_img, "mask":sli_lab}
            new_name = os.path.join(to_dir, basename.format("pos-"+str(zz)))
            pickle.dump(file_dict, open(new_name, "wb"))
        for zz in range(neg_img.shape[0]):
            sli_img = neg_img[zz, ...]
            sli_lab = neg_lab[zz, ...]
            file_dict = {"image":sli_img, "mask":sli_lab}
            new_name = os.path.join(to_dir, basename.format("neg-"+str(zz)))
            pickle.dump(file_dict, open(new_name, "wb"))
    except Exception as erro:
        wrong += 1
        if isprint: print(erro); print("***"*20)
    if isprint: print("All Wrongs: {}".format(wrong))
        
        
## make block_pkl for 3D segmentation  
def extract_block_1ch(info, to_dir, sub_dir="new_test", \
                min_ct_1ch=-1200, max_ct_1ch=800,             \
                pro_fn= None,         \
                input_depth=8,     step_size=4,      \
                phase="xiamen", tofor="disease", isprint=False):
    
    assert tofor in ["disease", "lobe"], "tofor should be disease or lobe"
    if tofor == "disease":
        assert phase in ["xiamen", "301", "covid", "others"], "phase should be xiamen, 301, or covid"
    if tofor == "lobe":
        assert phase in ["label",  "others"], "phase should be xiamen, 301, covid, or luna"
    wrong = 0
    to_dir = os.path.join(to_dir, sub_dir) 
    if not os.path.exists(to_dir): os.makedirs(to_dir)
    img_file, lab_file = info[:2]
    name_list = img_file.split("/")
    basename = "_".join([*name_list[-6:-1], name_list[-1][:-7].replace(" ", "").replace("-", ""), "{}.pkl"]).replace(" ", "-")
    try:
        if isprint: print(basename)
        img_ori = sitk.ReadImage(img_file, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img_ori)
        lab_ori = sitk.ReadImage(lab_file, sitk.sitkUInt16)
        lab = sitk.GetArrayFromImage(lab_ori)
        if pro_fn:
            img, lab = pro_fn(img, lab, lab_ori)
        ## normalization
        img_1ch = np.clip(img, min_ct_1ch, max_ct_1ch)
        img_1ch = (img_1ch-np.amin(img_1ch)+EPS)/(np.amax(img_1ch)-np.amin(img_1ch)+EPS)
        
        img = np.expand_dims(img_1ch, axis=-1) # shape:d, h, w, c
        ## ordering the label for disease of various hospital
        if tofor == "disease":
            if phase=="xiamen":
                lab[(lab>71)*(lab<78)] = 71
                lab[lab!=71] = 0
                lab[lab==71] = 1
            elif phase=="301":
                lab[lab!=71] = 0
                lab[lab==71] = 1
            elif phase=="covid":
                lab[(lab>194)*(lab<300)] = 71
                lab[lab!=71] = 0
                lab[lab==71] = 1
            else:
                pos         = (lab>=1)
                lab[pos]    = 1
                lab[~pos]   = 0
        elif tofor == "lobe":
            if phase=="label":
                #lab[(lab<219)+(lab>223)] = 0
                pos = ((lab>0)*(lab<=5)) + ((lab<=224)*(lab>=219))
                lab[~pos] = 0
                lab[lab==219] = 1
                lab[lab==220] = 2
                lab[lab==221] = 3
                lab[lab==222] = 4
                lab[lab==223] = 5
            elif phase=="others":
                pass
        pos_mask, neg_mask = None, None
        img = get_patches(img, input_depth=input_depth, step_size=step_size, dtype=np.float32)
        lab = get_patches(lab, input_depth=input_depth, step_size=step_size, dtype=np.uint8)
        if tofor == "disease":
            pos_mask = (lab == 1).any(axis=(-1, -2, -3))
        elif tofor == "lobe":
            pos_mask = (lab > 0).any(axis=(-1, -2, -3))
        neg_mask = ~pos_mask
        if isprint: print(">>>", "/".join(name_list[-4:-1]), ": ")
        pos_img = img[pos_mask]
        pos_lab = lab[pos_mask]
        if isprint: print("positive: ", pos_img.shape, end="  ")
        neg_img = img[neg_mask]
        neg_lab = lab[neg_mask]
        if isprint: print("negative: ", neg_img.shape )
        for zz in range(pos_img.shape[0]):
            sli_img = pos_img[zz, ...]
            sli_lab = pos_lab[zz, ...]
            file_dict = {"image":sli_img, "mask":sli_lab}
            new_name = os.path.join(to_dir, basename.format("pos-"+str(zz)))
            pickle.dump(file_dict, open(new_name, "wb"))
        for zz in range(neg_img.shape[0]):
            sli_img = neg_img[zz, ...]
            sli_lab = neg_lab[zz, ...]
            file_dict = {"image":sli_img, "mask":sli_lab}
            new_name = os.path.join(to_dir, basename.format("neg-"+str(zz)))
            pickle.dump(file_dict, open(new_name, "wb"))
    except Exception as erro:
        wrong += 1
        if isprint: print(erro); print("***"*20)
    if isprint: print("All Wrongs: {}".format(wrong))

def extract_block_2ch(info, to_dir, sub_dir="new_test", \
                min_ct_1ch=-1200, max_ct_1ch=800,             \
                min_ct_2ch=-1000,  max_ct_2ch=-100,            \
                pro_fn= None,         \
                input_depth=8,     step_size=4,      \
                phase="xiamen", tofor="disease", isprint=False):
    
    assert tofor in ["disease", "lobe"], "tofor should be disease or lobe"
    if tofor == "disease":
        assert phase in ["xiamen", "301", "covid", "others"], "phase should be xiamen, 301, or covid"
    if tofor == "lobe":
        assert phase in ["label",  "others"], "phase should be xiamen, 301, covid, or luna"
    wrong = 0
    to_dir = os.path.join(to_dir, sub_dir) 
    if not os.path.exists(to_dir): os.makedirs(to_dir)
    img_file, lab_file = info[:2]
    name_list = img_file.split("/")
    basename = "_".join([*name_list[-6:-1], name_list[-1][:-7].replace(" ", "").replace("-", ""), "{}.pkl"]).replace(" ", "-")
    try:
        if isprint: print(basename)
        img_ori = sitk.ReadImage(img_file, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img_ori)
        lab_ori = sitk.ReadImage(lab_file, sitk.sitkUInt16)
        lab = sitk.GetArrayFromImage(lab_ori)
        if pro_fn:
            img, lab = pro_fn(img, lab, lab_ori)
        ## normalization
        img_1ch = np.clip(img, min_ct_1ch, max_ct_1ch)
        img_1ch = (img_1ch-np.amin(img_1ch)+EPS)/(np.amax(img_1ch)-np.amin(img_1ch)+EPS)
        
        img_2ch = np.clip(img, min_ct_2ch, max_ct_2ch)
        img_2ch = (img_2ch-np.amin(img_2ch)+EPS)/(np.amax(img_2ch)-np.amin(img_2ch)+EPS)
        
        img = np.stack([img_1ch, img_2ch], axis=-1) # shape:d, h, w, c
        
        ## ordering the label for disease of various hospital
        if tofor == "disease":
            if phase=="xiamen":
                lab[(lab>71)*(lab<78)] = 71
                lab[lab!=71] = 0
                lab[lab==71] = 1
            elif phase=="301":
                lab[lab!=71] = 0
                lab[lab==71] = 1
            elif phase=="covid":
                lab[(lab>194)*(lab<300)] = 71
                lab[lab!=71] = 0
                lab[lab==71] = 1
            else:
                pos         = (lab>=1)
                lab[pos]    = 1
                lab[~pos]   = 0
        elif tofor == "lobe":
            if phase=="label":
                #lab[(lab<219)+(lab>223)] = 0
                pos = ((lab>0)*(lab<=5)) + ((lab<=224)*(lab>=219))
                lab[~pos] = 0
                lab[lab==219] = 1
                lab[lab==220] = 2
                lab[lab==221] = 3
                lab[lab==222] = 4
                lab[lab==223] = 5
            elif phase=="others":
                pass
        pos_mask, neg_mask = None, None
        img = get_patches(img, input_depth=input_depth, step_size=step_size, dtype=np.float32)
        lab = get_patches(lab, input_depth=input_depth, step_size=step_size, dtype=np.uint8)
        if tofor == "disease":
            pos_mask = (lab == 1).any(axis=(-1, -2, -3))
        elif tofor == "lobe":
            pos_mask = (lab > 0).any(axis=(-1, -2, -3))
        neg_mask = ~pos_mask
        if isprint: print(">>>", "/".join(name_list[-4:-1]), ": ")
        pos_img = img[pos_mask]
        pos_lab = lab[pos_mask]
        if isprint: print("positive: ", pos_img.shape, end="  ")
        neg_img = img[neg_mask]
        neg_lab = lab[neg_mask]
        if isprint: print("negative: ", neg_img.shape )
        for zz in range(pos_img.shape[0]):
            sli_img = pos_img[zz, ...]
            sli_lab = pos_lab[zz, ...]
            file_dict = {"image":sli_img, "mask":sli_lab}
            new_name = os.path.join(to_dir, basename.format("pos-"+str(zz)))
            pickle.dump(file_dict, open(new_name, "wb"))
        for zz in range(neg_img.shape[0]):
            sli_img = neg_img[zz, ...]
            sli_lab = neg_lab[zz, ...]
            file_dict = {"image":sli_img, "mask":sli_lab}
            new_name = os.path.join(to_dir, basename.format("neg-"+str(zz)))
            pickle.dump(file_dict, open(new_name, "wb"))
    except Exception as erro:
        wrong += 1
        if isprint: print(erro); print("***"*20)
    if isprint: print("All Wrongs: {}".format(wrong))
        

        
## DataSet and DataLoader        
class BaseDataset():
    def __init__(self, pos_paths, neg_paths=[], img_size=(256, 256), choice="all",
                    image_key="image", mask_key="mask",
                    preprocess=None, augmentation=None, proprocess=None,
                    seed=999, shuffle=True, pos_neg_ratio=4):
        assert choice.upper() in ["POS", "NEG", "ALL"], "Check The Parameter Choice"
        if shuffle:
            np.random.seed(seed)
        self.pos_paths      = pos_paths
        self.neg_paths = np.random.permutation(neg_paths).tolist() if shuffle else neg_paths
        paths = self.pos_paths + self.neg_paths[:len(pos_paths) // pos_neg_ratio]
        self.paths = np.random.permutation(paths).tolist() if shuffle else paths
        self.pos_length     = len(self.pos_paths)  
        self.neg_length     = len(self.neg_paths)
        self.length         = len(self.paths)

        self.img_size       = img_size
        self.choice         = choice
        self.image_key      = image_key
        self.mask_key       = mask_key
        self.augmentation   = augmentation
        self.preprocess     = preprocess
        self.proprocess     = proprocess
        self.seed           = seed
        np.random.seed(self.seed)
        if choice.upper() in ["NEG", "POS"]:
            if choice.upper() == "POS":
                self.paths = self.pos_paths 
            elif choice.upper() == "NEG":
                self.paths = self.neg_paths
    def _get_data(self, i):
        file_dict = pickle.load(open(self.paths[i], "rb"))
        img = file_dict[self.image_key]; lab = file_dict[self.mask_key]
        # print(self.paths[i])
        # if "neg" in self.paths[i]:
        #     lab = np.expand_dims(lab, -1)

        # img = cv2.resize(img_, self.img_size, interpolation=cv2.INTER_LINEAR)
        # lab = cv2.resize(lab_, self.img_size, interpolation=cv2.INTER_NEAREST)

        # assert img.shape == lab.shape, f"Assure The Same Shape.\nImage: {img.shape}, Anno: {lab.shape}"
        
        # if self.preprocess:
        #     if False: print("do preprocess")
        #     sample = self.preprocess(image=img, mask=lab)
        #     img, lab = sample["image"], sample["mask"]
        # if self.augmentation:
        #     sample = self.augmentation(image=img, mask=lab)
        #     img, lab= sample["image"], sample["mask"]
        # if self.proprocess:
        #     if False: print("do proprocess")
        #     sample = self.proprocess(image=img, mask=lab)
        #     img, lab = sample["image"], sample["mask"]

        return img.astype(np.float32), lab.astype(np.uint8)

    def __getitem__(self, i):
        return self._get_data(i)

    def __len__(self):
        return len(self.paths)
    
    def __add__(self, others):
        pos_paths_ = self.pos_paths + others.pos_paths
        neg_paths_ = self.neg_paths + others.neg_paths
        return self.__class__(pos_paths_, neg_paths_, self.img_size, self.choice, 
                              self.image_key, self.mask_key, 
                              self.augmentation, self.preprocess, self.proprocess,
                              self.seed,
                            )

    def __sub__(self, others):
        pos_paths_ = [path for path in self.pos_paths if path not in others.pos_paths] 
        neg_paths_ = [path for path in self.neg_paths if path not in others.neg_paths] 
        return self.__class__(pos_paths_, neg_paths_, self.img_size, self.choice, 
                              self.image_key, self.mask_key, 
                              self.augmentation, self.preprocess, self.proprocess,
                              self.seed,
                            )
    def shuffle_(self, factor=1.):
        if self.choice.upper()=="ALL":
            self.paths = self.pos_paths + np.random.permutation(self.neg_paths).tolist()[:int(self.neg_length*factor)]
            self.paths = np.random.permutation(self.paths).tolist()
        elif self.choice.upper() == "POS":
            self.paths = np.random.permutation(self.pos_paths).tolist()[:int(self.pos_length*factor)]
        elif self.choice.upper() == "NEG":
            self.paths = np.random.permutation(self.neg_paths).tolist()[:int(self.neg_length*factor)]

class Dataset3D(BaseDataset):

    def _get_data(self, i):
        file_dict = pickle.load(open(self.paths[i], "rb"))
        img_ = file_dict[self.image_key]; lab_ = file_dict[self.mask_key]
        first_rate = float(self.img_size[0])/float(img_.shape[1]) 
        secod_rate = float(self.img_size[1])/float(img_.shape[2])
        img = scipy.ndimage.zoom(img_, zoom=[1., first_rate, secod_rate, 1.], order=1) 
        if lab_.ndim == 4:
            lab = scipy.ndimage.zoom(lab_, zoom=[1., first_rate, secod_rate, 1.], order=0) 
        if lab_.ndim == 3:
            lab = scipy.ndimage.zoom(lab_, zoom=[1., first_rate, secod_rate], order=0)
#         assert img.shape == lab.shape, "Assure The Same Shape"
        if self.preprocess:
            if False: print("do preprocess")
            sample = self.preprocess(image=img, mask=lab)
            img, lab = sample["image"], sample["mask"]
        if self.augmentation:
            sample = self.augmentation(image=img, mask=lab)
            img, lab= sample["image"], sample["mask"]
        if self.proprocess:
            if False: print("do proprocess")
            sample = self.proprocess(image=img, mask=lab)
            img, lab = sample["image"], sample["mask"]
        if img.ndim == 3: img = np.expand_dims(img, axis=-1); lab = np.expand_dims(lab, axis=-1)

        return img.astype(np.float32), lab.astype(np.uint8)

class KerasLoader(keras.utils.Sequence):
    def __init__(self, dataset, factor=1., batch_size=32, shuffle=False):
        assert factor <= 1. and factor > 0, "Please Ensure The Factor"
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.factor     = factor 

        ## caculate the right length of dataset in different phase 
        if   self.dataset.choice.upper() == "ALL":
            self.length_    = self.dataset.pos_length + int(self.factor*self.dataset.neg_length)
        elif self.dataset.choice.upper() == "POS":
            self.length_    = int(self.dataset.pos_length*self.factor)
        elif self.dataset.choice.upper() == "NEG":
            self.length_    = int(self.dataset.neg_length*self.factor)

        self.indexes = np.arange(self.length_)
        self.on_epoch_end()
    def __getitem__(self, i):
        start = i* self.batch_size
        stop = (i+1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[self.indexes[j]])
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch
    def __len__(self):
        return len(self.indexes)//self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.dataset.shuffle_(self.factor)
            self.indexes = np.random.permutation(self.indexes)
    
