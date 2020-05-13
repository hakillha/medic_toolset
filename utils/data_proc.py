import os
from os.path import join as pj

import numpy as np
import SimpleITK as sitk

EPS = 1.0e-8

# make slice_pkl for 2D segmentation   
def extract_slice(info, out_dir, debug, min_ct=-1400, max_ct=800):
    erroneous_cnt = 0
    im_file, ann_file = info
    out_file = os.path.basename(im_file)
    # try:
    if debug: 
        print(out_file)
    im_itk = sitk.ReadImage(im_file, sitk.sitkFloat32)
    im = sitk.GetArrayFromImage(im_itk)
    ann_itk = sitk.ReadImage(ann_file, sitk.sitkUInt16)
    ann = sitk.GetArrayFromImage(ann_itk)
    # normalization
    im = np.clip(im, min_ct, max_ct) # Value interval of CT is different from photos
    im = (im - np.amin(im) + EPS) / (np.amax(im) - np.amin(im) + EPS)
    # depth, h, w, batch_size?
    im = np.expand_dims(im, axis=-1) # shape:d, h, w, c
    ann[ann >= 1], ann[ann < 1] = 1, 0
    annotated_depth_mask = (ann == 1).any(axis=(-1, -2))
    if debug: 
        print(">>>", out_file)
    annotated_im, annotated_ann = im[annotated_depth_mask], ann[annotated_depth_mask]
    if debug: 
        print("Positive: ", annotated_im.shape)
    unannotated_im, unannotated_ann = im[~annotated_depth_mask], ann[~annotated_depth_mask]
    if debug: 
        print("Negative: ", unannotated_im.shape)
    for i in range(annotated_im.shape[0]):
        im_slice = annotated_im[i, ...]
        ann_slice = annotated_ann[i, ...]
        # file_dict = {"image":sli_img, "mask":sli_lab}
        # new_name = pj(out_dir, basename.format("pos-"+str(zz)))
        # pickle.dump(file_dict, open(new_name, "wb"))
    for i in range(unannotated_im.shape[0]):
        im_slice = unannotated_im[i, ...]
        ann_slice = unannotated_ann[i, ...]
        # file_dict = {"image":sli_img, "mask":sli_lab}
        # new_name = pj(out_dir, basename.format("neg-"+str(zz)))
        # pickle.dump(file_dict, open(new_name, "wb"))
    # except Exception as erro:
    #     erroneous_cnt += 1
    #     if debug: 
    #         print(erro)
    #         print("***"*20)
    if debug: 
        print(f"Number of erroneous data files: {erroneous_cnt}")