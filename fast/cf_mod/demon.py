import argparse, os, sys
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from os.path import join as pj

import SimpleITK as sitk

# sys.path.insert(0, "/rdfs/fast/home/sunyingge/code/chenfeng")
from misc.utils import get_infos
from misc.data_tools import extract_slice_1ch

def get_slice_dataset(
        data_dir,
        data_dir_post,
        out_dir,
        phase="others",
        link="-",
        pro_fn=None,
        isprint=False
        ):
    info_paths = get_infos(pj(data_dir, data_dir_post), link=link)
    target_dir = pj(out_dir, data_dir_post)
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
    func_1 = partial(extract_slice_1ch,
                    to_dir=out_dir, sub_dir=data_dir_post,
                    min_ct_1ch=-1400, max_ct_1ch=800,
                    pro_fn=None,
                    tofor="disease", phase=phase, isprint=isprint)
    pool = mp.Pool(8)
    pool.map(func_1, info_paths)
    pool.close() 

def collect_stats(full_data_dir, thickness_thres):
    if not os.path.exists(full_data_dir):
        print(f"{full_data_dir} doesn't exist!")
        return
    stats = defaultdict(int)
    for root, _, files in os.walk(full_data_dir):
        nii_cnt = len([f for f in files if f.endswith(".nii.gz")])
        if nii_cnt > 2:
            stats["Num of nii>2"] += 1
            continue
        else:
            for f in files:
                # This however doesn't deal with the case that both nii files are im
                # files while it does throw away cases where both are anno files
                if not f.endswith("label.nii.gz") and f.endswith(".nii.gz"):
                    im_nii = sitk.ReadImage(pj(root, f))
                    thickness = round(im_nii.GetSpacing()[-1], 1)
                    if 'covid_pneu_datasets' in root:
                        if thickness >= thickness_thres:
                            stats['covid'] += 1
                            stats['thick'] += 1
                            stats['covid_thick'] += 1
                        elif thickness < thickness_thres:
                            stats['covid'] += 1
                            stats['thin'] += 1
                            stats['covid_thin'] += 1
                    elif 'normal_pneu_datasets' in root:
                        if thickness >= thickness_thres:
                            stats['normal'] += 1
                            stats['thick'] += 1
                            stats['normal_thick'] += 1
                        elif thickness < thickness_thres:
                            stats['normal'] += 1
                            stats['thin'] += 1
                            stats['normal_thin'] += 1
                    else:
                        stats["Not in covid nor normal_pneu"] += 1
    return stats

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_dir",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/0508/Train",
        default="/rdfs/fast/home/sunyingge/data/COV_19/0508/Val/"
        )
    parser.add_argument("--data_dir_postfix", nargs='+', 
        help="""This is added to the ends of both '--data_dir' and
        '--out_dir'.""",
        # default=["batch_0505", "batch_0508", "batch_0509", "batch_0510", "batch_0511", 
        #     "batch_0513", "batch_0514", "batch_0515"],
        default=["batch_0505", "batch_0507", "batch_0508", "batch_0509", "batch_0510", 
            "batch_0511", "batch_0513", "batch_0514", "batch_0515"]
            )
    parser.add_argument("--out_dir",
        # default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0516")
    )
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--thickness_thres", default=3.0)
    return parser.parse_args()

args = parse_args()

for post in args.data_dir_postfix:
    stats_res = collect_stats(pj(args.data_dir, post), args.thickness_thres)
    print(f"{post}: ")
    print(stats_res)
    if not args.stats:
        get_slice_dataset(args.data_dir, post, args.out_dir)