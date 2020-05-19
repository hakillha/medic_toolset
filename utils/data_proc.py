import glob, os
from os.path import join as pj

import numpy as np
import pickle
import SimpleITK as sitk

EPS = 1.0e-8

# make slice_pkl for 2D segmentation   
# def extract_slice(info, out_dir, debug, min_ct=-1400, max_ct=800):
#     erroneous_cnt = 0
#     im_file, ann_file = info
#     out_file = os.path.basename(im_file)
#     # try:
#     if debug: 
#         print(out_file)
#     im_itk = sitk.ReadImage(im_file, sitk.sitkFloat32)
#     im = sitk.GetArrayFromImage(im_itk)
#     ann_itk = sitk.ReadImage(ann_file, sitk.sitkUInt16)
#     ann = sitk.GetArrayFromImage(ann_itk)
#     # normalization
#     im = np.clip(im, min_ct, max_ct) # Value interval of CT is different from photos
#     im = (im - np.amin(im) + EPS) / (np.amax(im) - np.amin(im) + EPS)
#     # depth, h, w, batch_size?
#     im = np.expand_dims(im, axis=-1) # shape:d, h, w, c
#     ann[ann >= 1], ann[ann < 1] = 1, 0
#     annotated_depth_mask = (ann == 1).any(axis=(-1, -2))
#     if debug: 
#         print(">>>", out_file)
#     annotated_im, annotated_ann = im[annotated_depth_mask], ann[annotated_depth_mask]
#     if debug: 
#         print("Positive: ", annotated_im.shape)
#     unannotated_im, unannotated_ann = im[~annotated_depth_mask], ann[~annotated_depth_mask]
#     if debug: 
#         print("Negative: ", unannotated_im.shape)
#     for i in range(annotated_im.shape[0]):
#         im_slice = annotated_im[i, ...]
#         ann_slice = annotated_ann[i, ...]
#         # file_dict = {"image":sli_img, "mask":sli_lab}
#         # new_name = pj(out_dir, basename.format("pos-"+str(zz)))
#         # pickle.dump(file_dict, open(new_name, "wb"))
#     for i in range(unannotated_im.shape[0]):
#         im_slice = unannotated_im[i, ...]
#         ann_slice = unannotated_ann[i, ...]
#         # file_dict = {"image":sli_img, "mask":sli_lab}
#         # new_name = pj(out_dir, basename.format("neg-"+str(zz)))
#         # pickle.dump(file_dict, open(new_name, "wb"))
#     # except Exception as erro:
#     #     erroneous_cnt += 1
#     #     if debug: 
#     #         print(erro)
#     #         print("***"*20)
#     if debug: 
#         print(f"Number of erroneous data files: {erroneous_cnt}")

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
        multicat, discard_neg):
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
        im_np = np.clip(im_np, min_ct_1ch, max_ct_1ch)
        EPS = 1e-8
        im_np = (im_np - np.amin(im_np) + EPS) / (np.amax(im_np) - np.amin(im_np) + EPS)
        im_np = np.expand_dims(im_np, -1)
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