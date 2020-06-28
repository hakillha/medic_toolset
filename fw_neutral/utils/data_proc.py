import csv, glob, os, pickle, random
random.seed(123)
from collections import defaultdict
from os.path import join as pj
from pprint import pprint

import cv2
import numpy as np
import SimpleITK as sitk

import sys
sys.path.insert(0, "../..")
from fw_neutral.utils.metrics import Patient, Pneu_type

EPS = 1.0e-8

def ID_quality_map(quality_file, quality_list):
    quality_filter = defaultdict(bool)
    num_quality_p = 0
    with open(quality_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            for quality in quality_list:
                if row[1] == quality:
                    quality_filter[row[0]] = True
                    num_quality_p += 1
    return quality_filter, num_quality_p

def MD5_datadir_map(fmap_file):
    md5_map = {}
    with open(fmap_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            md5_map[Patient.find_id(row[1])] = row[0]
    return md5_map

def Combine_pndirs(pn_ratio, pdirs, ndirs):
    if pn_ratio:
        if pn_ratio != "all":
            if pn_ratio > 1:
                ndirs = np.random.permutation(ndirs).tolist()[:len(pdirs) // pn_ratio]
            else:
                pdirs = np.random.permutation(pdirs).tolist()[:int(len(ndirs) * pn_ratio)]
        print(f"\nNum of pos samples: {len(pdirs)}")
        print(f"Num of neg samples: {len(ndirs)}\n")
        print(f"Actual ratio: {len(pdirs)/len(ndirs)}(p2n)")
        if len(pdirs) >= len(ndirs):
            list1 = pdirs
            list2 = ndirs  
        else:
            list1 = ndirs
            list2 = pdirs
        posis1 = True if len(pdirs) >= len(ndirs) else False
        ratio = len(list1) / len(list2)
        incl_prob = ratio - int(ratio)
        print((ratio, incl_prob))
        res, i, j = [], 0, 0
        while j < len(list2) and i + int(ratio) + 1 < len(list1):
            res.append(list2[j])
            j += 1
            num_samp = int(ratio) + 1 if np.random.uniform() < incl_prob else int(ratio)
            res += list1[i:i + num_samp]
            i += num_samp
        left1, left2 = len(list1) - i, len(list2) - j
        if posis1:
            leftset = "pos" if left1 > left2 else "neg"
        else:
            leftset = "neg" if left1 > left2 else "pos"
        print(f"Num of {leftset} samples left: {max(left1, left2)}")
        return res
    else:
        return pdirs

def Patient_quality_filter(in_data_dirs, in_p_ids, fmap_file, quality_file, quality_list):
    md5_q_map, num_quality_p = ID_quality_map(quality_file, quality_list)
    md5_map = MD5_datadir_map(fmap_file)

    if not in_data_dirs is None:
        res, p_id_set = [], set()
        matched_patient_set = set()
        # patientsfoundinmap = set()
        for data_dir in in_data_dirs:
            p_id = Patient.find_id(data_dir)
            p_id_set.add(p_id)
            if p_id in md5_map.keys():
                # patientsfoundinmap.add(p_id)
                if md5_q_map[md5_map[p_id]]:
                    matched_patient_set.add(p_id)
                    res.append(data_dir)
        num_matches = len(matched_patient_set)
    elif not in_p_ids is None:
        filtered_id_list = []
        for p_id in in_p_ids:
            if p_id in md5_map.keys() and md5_q_map[md5_map[p_id]]:
                filtered_id_list.append(p_id)
        num_matches = len(filtered_id_list)
        res = filtered_id_list
    
    # print #all patients
    print(f"Number of quality patients: {num_quality_p}")
    print(f"Number of matched patients: {num_matches}")
    # print(len(patientsfoundinmap))
    # return res, list(p_id_set)
    return res

def find_positivity(data_dir):
    if "pos" in data_dir:
        return "pos"
    elif "neg" in data_dir:
        return "neg"

def gen_data_list(data_dir, cfg):
    data_dirs = paths_from_data(data_dir, None, "all")
    p_slice_map = defaultdict(lambda : {"pos": [], "neg": []})
    for data_dir in data_dirs:
        p_slice_map[Patient.find_id(data_dir)][find_positivity(data_dir)].append(data_dir)
    print(f"Num of patients: {len(p_slice_map.keys())}")
    filtered_ids = Patient_quality_filter(None, p_slice_map.keys(), 
        cfg.trainset["md5_map"], cfg.trainset["patient_filter_file"], cfg.trainset["quality"])
    # random.shuffle(filtered_ids)
    filtered_ids = np.random.permutation(filtered_ids).tolist()
    val_ids = filtered_ids[-int(len(filtered_ids) * cfg.trainset["val_ratio"]):]
    train_ids = filtered_ids[:len(filtered_ids) - len(val_ids)]
    train_pdirs, train_ndirs, val_dirs = [], [], []
    for idx in train_ids:
        train_pdirs += p_slice_map[idx]["pos"]
        train_ndirs += p_slice_map[idx]["neg"]
    train_dirs = Combine_pndirs(cfg.trainset["pn_ratio"], train_pdirs, train_ndirs)
    for idx in val_ids:
        val_dirs += (p_slice_map[idx]["pos"] + p_slice_map[idx]["neg"])
    print(f"Num of training samples: {len(train_dirs)}")
    print(f"Num of validation samples: {len(val_dirs)}")
    return train_dirs, val_dirs

# def get_infos(root_dir, link="-", isprint=True):
#     get_paths = []
#     # get_num = 0
#     for root, dirs, files_ in os.walk(root_dir, topdown=True):
#         files = []
#         for idx in files_:
#             if idx.endswith(".nii.gz"): files.append(idx)
#         if len(files)%2==0:
#             if len(files)>2:
#                 img_file, lab_file = "", ""
#                 for inx in files:
#                     if inx.endswith(f"{link}label.nii.gz"):
#                         lab_file = os.path.join(root, inx)
#                         img_file = lab_file.replace(f"{link}label.nii.gz", ".nii.gz")
#                         if os.path.exists(img_file):
#                             get_num += 1
#                             get_paths.append([img_file, lab_file])
#                         else:
#                             print(root)
#             if len(files) == 2:
#                 # img_file, lab_file = "", ""
#                 for inx in files:
#                     if inx.endswith(f"{link}label.nii.gz"):
#                         lab_file = os.path.join(root, inx)
#                     elif inx.endswith(".nii.gz"):
#                         img_file = os.path.join(root, inx)
#                 # if len(img_file) and len(lab_file):
#                 #     get_num += 1
#                 #     get_paths.append([img_file, lab_file]) 
#                 get_paths.append([img_file, lab_file]) 
#     if isprint: print("all_picked: {}".format(len(get_paths)))
#     return get_paths 

def get_infos(root_dir, output=False):
    """
        args:
            output: For an output dir there can be
                3 nii files at each leaf dir.
    """
    assert os.path.exists(root_dir), "Data directory doesn't exist!"
    sample_list = []
    for root, dirs, files in os.walk(root_dir):
        data_fs = []
        for f in files:
            if f.endswith(".nii.gz"):
                data_fs.append(f)
        for f in data_fs:
            # It's checked that all anno files end with this suffix
            if f.endswith("label.nii.gz"):
                gt_file = pj(root, f)
            elif f.endswith("_pred.nii.gz"):
                res_file = pj(root, f)
            else:
                im_file = pj(root, f)
        if len(data_fs):
            if output:
                assert len(data_fs) == 3, root
                sample_list.append([im_file, gt_file, res_file])
            else:
                assert len(data_fs) == 2, root
                sample_list.append([im_file, gt_file])
    print(f"Num of samples: {len(sample_list)}")
    return sample_list

def extract_slice_single_file(info_path, out_dir, min_ct_1ch, max_ct_1ch, multicat, 
    discard_neg, norm_by_interval, include_healthy, thickness_thres):
    im_file, anno_file = info_path
    stats = defaultdict(int)
    condition_cat = Pneu_type(im_file, include_healthy, discard_neg)
    if condition_cat == None:
        return None, None
    elif condition_cat == "common_pneu":
        condition_cat = "normal_pneu"
    try:
        # patient_id = im_file.split('/')[-2]
        patient_id = Patient.find_id(im_file)
        im_nii = sitk.ReadImage(im_file, sitk.sitkFloat32)
        thickness = "thick" if im_nii.GetSpacing()[-1] >= thickness_thres else "thin"
        im_np = sitk.GetArrayFromImage(im_nii)
        ann_np = sitk.GetArrayFromImage(sitk.ReadImage(anno_file, sitk.sitkUInt16))
        # im_np = im_normalize(im_np, [min_ct_1ch, max_ct_1ch], norm_by_interval)
        pos_index = ann_np >= 1
        if multicat:
            if condition_cat == "normal_pneu":
                ann_np[pos_index] = 2
            elif condition_cat == "covid_pneu":
                ann_np[pos_index] = 1
            ann_np[~pos_index] = 0
        else:
            ann_np[pos_index], ann_np[~pos_index] = 1, 0

        # separate the pos/neg slices
        pos_slice_mask = (ann_np > 0).any(axis=(-1, -2))
        pos_im, pos_ann = im_np[pos_slice_mask], ann_np[pos_slice_mask]
        for sli in range(pos_im.shape[0]):
            save_dir = pj(out_dir, '_'.join([condition_cat, thickness]), patient_id, 'pos')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(pj(save_dir, f"{sli}.pkl"), "wb") as f:
                pickle.dump({"im": pos_im[sli, ...], "mask": pos_ann[sli, ...]}, f)

        if not discard_neg or (include_healthy and condition_cat == "healthy"):
            neg_slice_mask = ~pos_slice_mask
            neg_im, neg_ann = im_np[neg_slice_mask], ann_np[neg_slice_mask]
            for sli in range(neg_im.shape[0]):
                save_dir = pj(out_dir, '_'.join([condition_cat, thickness]), patient_id, 'neg')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(pj(save_dir, f"{sli}.pkl"), "wb") as f:
                    pickle.dump({"im": neg_im[sli, ...], "mask": neg_ann[sli, ...]}, f)
        
        stats[condition_cat] = 1
        stats[thickness] = 1
        stats[condition_cat + '_' + thickness] = 1
        stats[condition_cat + '_pos_slices'] = pos_im.shape[0]
        stats[condition_cat + '_neg_slices'] = im_np.shape[0] - pos_im.shape[0]
        return im_file, stats
    except Exception as error:
        print(error)
        return None, None

def extract_slice_sequential(info_paths, out_dir, data_dir_suffix, min_ct_1ch, max_ct_1ch):
    for im_file, anno_file in info_paths:
        print(im_file)
        extract_slice_single_file((im_file, anno_file), out_dir, data_dir_suffix, min_ct_1ch, max_ct_1ch)
        
def paths_from_data(data_dir, datalist, sample_set):
    sample_paths = []
    if not data_dir is None:
        assert os.path.exists(data_dir)
        for root, dirs, files in os.walk(data_dir):
            if sample_set == "all":
                sample_paths += glob.glob(pj(root, "*.pkl"))
            elif sample_set in root:
                sample_paths += glob.glob(pj(root, "*.pkl"))
    if not datalist is None:
        for data_dir in datalist:
            if sample_set in data_dir:
                sample_paths.append(data_dir)
    return sample_paths

def im_normalize(im, ct_interval, norm_by_interval):
    EPS = 1e-8
    min_ct_1ch, max_ct_1ch = ct_interval
    im = np.clip(im, min_ct_1ch, max_ct_1ch)
    if norm_by_interval:
        im = (im - min_ct_1ch + EPS) / (max_ct_1ch - min_ct_1ch + EPS)
    else:
        im = (im - np.amin(im) + EPS) / (np.amax(im) - np.amin(im) + EPS)
    im = np.expand_dims(im, -1)
    return im

class extra_processing():
    def __init__(self, im_size, num_class, preproc_cfg, loss_type, og_shape=None):
        self.im_size = im_size
        self.num_class = num_class
        self.cfg = preproc_cfg
        self.loss = loss_type
        self.tl_list = []
        self.og_shape = og_shape

    def preprocess(self, im, anno, training=True):
        """
            args:
                im: [h, w]
        """
        if self.num_class == 1:
            # to make it compatible with mutlicls label
            anno = anno > 0 

        im = im_normalize(im, self.cfg["normalize"]["ct_interval"],
            self.cfg["normalize"]["norm_by_interval"])

        if training:
            # elif self.loss == "sigmoid":
            #     ann_ar = np.repeat(anno[:,1,:,:,:], self.cfg.num_class + 1, -1)
            #     all_cls_ids = np.ones(shape=ann_ar.shape)
            #     for i in range(self.cfg.num_class + 1):
            #         all_cls_ids[...,i] = all_cls_ids[...,i] * i
            #     ann_ar = ann_ar == all_cls_ids
            if self.cfg["flip"]:
                flip_rate = .5
                if np.random.rand() < flip_rate:
                    im = np.flip(im, 1)
                    anno = np.flip(anno, 1)

        # Keep shape info in comments
        if self.cfg["cropping"]:
            im_above_thres = np.argwhere(im > self.cfg["cropping_ct_thres"])
            if len(im_above_thres):
                center_y, center_x, _ = map(int, np.mean(im_above_thres, axis=0))
            else:
                center_y, center_x = im.shape[0] // 2, im.shape[1] // 2
            if training:
                randomness = self.cfg["cropping_train_randomness"]
                center_y += random.randint(-randomness, randomness)
                center_x += random.randint(-randomness, randomness)
            x1, x2 = center_x - self.im_size[0] // 2, center_x + self.im_size[0] // 2
            y1, y2 = center_y - self.im_size[1] // 2, center_y + self.im_size[1] // 2
            if x1 < 0:
                x1, x2 = 0, self.im_size[0]
            if y1 < 0:
                y1, y2 = 0, self.im_size[1]
            if x2 > im.shape[1]:
                x1, x2 = im.shape[1] - self.im_size[0], im.shape[1]
            if y2 > im.shape[0]:
                y1, y2 = im.shape[0] - self.im_size[1], im.shape[0]
            im = im[y1:y2,x1:x2]
            if training:
                anno = anno[y1:y2,x1:x2]
            ret = [im, anno if self.loss == "softmax" else np.expand_dims(anno, -1)]
            return ret if training else ret + [(x1, y1)] # also need the tl coords during eval
        elif self.cfg["resize"]:
            im = cv2.resize(im, self.im_size, interpolation=cv2.INTER_LINEAR)
            if training:
                anno = cv2.resize(anno, self.im_size, interpolation=cv2.INTER_NEAREST)
            return np.expand_dims(im, -1), anno if self.loss == "softmax" else np.expand_dims(anno, -1)
    
    def batch_preprocess(self, im_batch, anno_batch, training):
        im_stack, anno_stack = [], []
        if training:
            pass
        else:
            for i in range(im_batch.shape[0]):
                res = self.preprocess(im_batch[i,:,:], anno_batch[i,:,:], training)
                if self.cfg["cropping"]:
                    im_stack.append(res[0])
                    anno_stack.append(res[1])
                    # Record the tl coordinates of the crops in order to "paste" them back
                    # We don't need to clear this list since every loop we create a
                    # new extraprocessing instance
                    self.tl_list.append(res[2])
                elif self.cfg["resize"]:
                    im_stack.append(res)
            return np.array(im_stack), np.array(anno_stack)

    def batch_postprocess(self, pred_batch):
        """
            returns:
                [C, H, W]
        """
        pred_stack = []
        assert len(self.tl_list) == pred_batch.shape[0], f"tl list length: {len(self.tl_list)}"
        for i in range(pred_batch.shape[0]):
            if self.cfg["cropping"]:
                padded_res = np.zeros(shape=self.og_shape)
                padded_res[self.tl_list[i][1]:self.tl_list[i][1] + self.im_size[0], \
                    self.tl_list[i][0]:self.tl_list[i][0] + self.im_size[1]] = pred_batch[i,:,:,0]
                pred_stack.append(padded_res)
            elif self.cfg["resize"]:
                pred_stack.append(cv2.resize(pred_batch[i,:,:,0].astype(np.float32), self.og_shape, interpolation=cv2.INTER_NEAREST))
        self.tl_list = [] # clear up this list
        return np.expand_dims(np.array(pred_stack), -1)

if __name__ == "__main__":
    Patient_quality_filter("/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/trainq.csv")