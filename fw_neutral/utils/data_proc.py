import glob, os
from os.path import join as pj
import random
random.seed(123)

import numpy as np
import pickle
import SimpleITK as sitk

EPS = 1.0e-8

def get_infos(root_dir, link="-", isprint=True):
    get_paths = []
    get_num = 0
    for root, dirs, files_ in os.walk(root_dir, topdown=True):
        files = []
        for idx in files_:
            if idx.endswith(".nii.gz"): files.append(idx)
        if len(files)%2==0:
            if len(files)>2:
                img_file, lab_file = "", ""
                for inx in files:
                    if inx.endswith(f"{link}label.nii.gz"):
                        lab_file = os.path.join(root, inx)
                        img_file = lab_file.replace(f"{link}label.nii.gz", ".nii.gz")
                        if os.path.exists(img_file):
                            get_num += 1
                            get_paths.append([img_file, lab_file])
                        else:
                            print(root)
            if len(files) == 2:
                img_file, lab_file = "", ""
                for inx in files:
                    if inx.endswith(f"{link}label.nii.gz"):
                        lab_file = os.path.join(root, inx)
                    elif inx.endswith(".nii.gz"):
                        img_file = os.path.join(root, inx)
                if len(img_file) and len(lab_file):
                    get_num += 1
                    get_paths.append([img_file, lab_file]) 
    if isprint: print("all_picked: {}".format(get_num))
    return get_paths 

def extract_slice_single_file(info_path, out_dir, data_dir_post, min_ct_1ch, max_ct_1ch,
        multicat, discard_neg, norm_by_interval):
    im_file, anno_file = info_path
    # print(im_file)
    if "covid_pneu" in im_file:
        condition_cat = "covid_pneu"
    elif "normal_pneu" in im_file:
        condition_cat = "normal_pneu"
    elif "healthy" in im_file:
        if discard_neg:
            return
        condition_cat = "healthy"
    else:
        print("Unknown condition!")
        return
    try:
        patient_id = im_file.split('/')[-2]
        im_np = sitk.GetArrayFromImage(sitk.ReadImage(im_file, sitk.sitkFloat32))
        ann_np = sitk.GetArrayFromImage(sitk.ReadImage(anno_file, sitk.sitkUInt16))
        im_np = im_normalize(im_np, min_ct_1ch, max_ct_1ch, norm_by_interval)
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
            save_dir = pj(out_dir, data_dir_post, condition_cat, patient_id, 'pos')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(pj(save_dir, f"{sli}.pkl"), "wb") as f:
                pickle.dump({"im": pos_im[sli, ...], "mask": pos_ann[sli, ...]}, f)

        if not discard_neg:
            neg_slice_mask = ~pos_slice_mask
            neg_im, neg_ann = im_np[neg_slice_mask], ann_np[neg_slice_mask]
            for sli in range(neg_im.shape[0]):
                save_dir = pj(out_dir, data_dir_post, condition_cat, patient_id, 'neg')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(pj(save_dir, f"{sli}.pkl"), "wb") as f:
                    pickle.dump({"im": neg_im[sli, ...], "mask": neg_ann[sli, ...]}, f)
    except Exception as error:
        print(error)

def extract_slice_sequential(info_paths, out_dir, data_dir_post, min_ct_1ch, max_ct_1ch):
    for im_file, anno_file in info_paths:
        print(im_file)
        extract_slice_single_file((im_file, anno_file), out_dir, data_dir_post, min_ct_1ch, max_ct_1ch)
        
def paths_from_data(data_dir):
    sample_paths = []
    for root, dirs, files in os.walk(data_dir):
        if "pos" in root:
            sample_paths += glob.glob(pj(root, "*.pkl"))
    return sample_paths

def im_normalize(im, min_ct_1ch, max_ct_1ch, norm_by_interval):
    EPS = 1e-8
    im = np.clip(im, min_ct_1ch, max_ct_1ch)
    if norm_by_interval:
        im = (im - min_ct_1ch + EPS) / (max_ct_1ch - min_ct_1ch + EPS)
    else:
        im = (im - np.amin(im) + EPS) / (np.amax(im) - np.amin(im) + EPS)
    im = np.expand_dims(im, -1)
    return im

def preprocess(im, anno, cfg, training):
    if cfg.preprocess["cropping"]:
        im_above_thres = np.argwhere(im > cfg.preprocess["cropping_ct_thres"])
        if len(im_above_thres):
            center_y, center_x, _ = map(int, np.mean(im_above_thres, axis=0))
        else:
            center_y, center_x = im.shape[0] // 2, im.shape[1] // 2
        if training:
            randomness = cfg.preprocess["cropping_train_randomness"]
            center_y += random.randint(-randomness, randomness)
            center_x += random.randint(-randomness, randomness)
        x1, x2 = center_x - cfg.im_size[0] // 2, center_x + cfg.im_size[0] // 2
        y1, y2 = center_y - cfg.im_size[1] // 2, center_y + cfg.im_size[1] // 2
        if x1 < 0:
            x1, x2 = 0, cfg.im_size[0]
        if y1 < 0:
            y1, y2 = 0, cfg.im_size[1]
        if x2 > im.shape[1]:
            x1, x2 = im.shape[1] - cfg.im_size[0], im.shape[1]
        if y2 > im.shape[0]:
            y1, y2 = im.shape[0] - cfg.im_size[1], im.shape[0]
        im = im[y1:y2,x1:x2]
        if training:
            anno = anno[y1:y2,x1:x2]
            return im, np.expand_dims(anno, -1)
        else:
            return im, (x1, y1)
    elif cfg.preprocess["resize"]:
        im = cv2.resize(im, cfg.im_size, interpolation=cv2.INTER_LINEAR)
        if training:
            anno = cv2.resize(anno, cfg.im_size, interpolation=cv2.INTER_NEAREST)
            return im, np.expand_dims(anno, -1)
        else:
            return im

# For evaluation
# def postprocess(pred, cfg, topleft_list=None):
#     if cfg.preprocess["cropping"]:
#         pass
#     elif cfg.preprocess["resize"]:
#         pass