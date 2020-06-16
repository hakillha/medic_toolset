import os, time
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pj

def viz_patient(im, output_mask, gt_mask, show, 
    alpha=.1, out_dir=None, im_fname=None):
    """
        im: [C, H, W]
    """
    if not show:
        out_dir = pj(out_dir, os.path.basename(im_fname).split('.')[0])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    plt.gcf().set_size_inches(14.0, 14.0)
    if len(im.shape) == 3:
        im = np.expand_dims(im, -1)
    if len(output_mask.shape) == 3:
        output_mask = np.expand_dims(output_mask, -1)
    if len(gt_mask.shape) == 3:
        gt_mask = np.expand_dims(gt_mask, -1)
    for i in range(im.shape[0]):
        im_slice = np.repeat(im[i], 3, -1)
        plt.imshow(im_slice)
        color = np.zeros_like(im_slice)
        color += np.array([[[0, 1, 0]]])
        plt.imshow(np.dstack([color, output_mask[i] * alpha]))
        color = np.zeros_like(im_slice)
        color += np.array([[[1, 0, 0]]])
        plt.imshow(np.dstack([color, gt_mask[i] * alpha]))
        if show:
            plt.show()
            # time.sleep(100)
            input("")
        else:
            plt.savefig(pj(out_dir, f"{i:04}.jpg"))
        plt.clf()