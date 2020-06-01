import os, time
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pj

def viz_patient(im, output_mask, gt_mask, out_dir, im_fname, alpha=.1):
    out_dir = pj(out_dir, os.path.basename(im_fname).split('.')[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.gcf().set_size_inches(14.0, 14.0)
    for i in range(im.shape[0]):
        im_slice = np.repeat(im[i], 3, -1)
        plt.imshow(im_slice)
        color = np.zeros_like(im_slice)
        color += np.array([[[0, 1, 0]]])
        plt.imshow(np.dstack([color, np.expand_dims(output_mask[i], -1) * alpha]))
        color = np.zeros_like(im_slice)
        color += np.array([[[1, 0, 0]]])
        plt.imshow(np.dstack([color, np.expand_dims(gt_mask[i], -1) * alpha]))
        plt.savefig(pj(out_dir, f"{i:04}.jpg"))
        # plt.show()
        # time.sleep(100)
        # input("")
        plt.clf()