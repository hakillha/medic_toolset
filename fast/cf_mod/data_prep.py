import os, sys
import multiprocessing as mp
from functools import partial
from os.path import join as pj

ip_dir = "/rdfs/fast/home/sunyingge/code/chenfeng"
sys.path.insert(0, ip_dir)
from misc.data_tools import extract_slice_1ch
from misc.utils import get_infos

def get_slice_dataset(
        root_dir="/rdfs/data/chenfeng/Lung/LungPneumonia/patients/train/healthy_datasets/",
        sub_dir="train/healthy_datasets",
        to_dir = "/rdfs/data/chenfeng/Lung/LungPneumonia/slices_1ch",
        phase="others",
        link="-",
        pro_fn=None,
        isprint=False
        ):
    info_paths = get_infos(root_dir, link=link)
    target_dir = os.path.join(to_dir, sub_dir)
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
        func_1 = partial(extract_slice_1ch,
                        to_dir=to_dir, sub_dir=sub_dir,         \
                        min_ct_1ch=-1400, max_ct_1ch=800,             \
                        pro_fn=None,                             \
                        tofor="disease", phase=phase, isprint=isprint)
        pool = mp.Pool(8)
        pool.map(func_1, info_paths)
        pool.close()
    else:
        print("Already generated, skipped...")

for split in ["Train/batch_0505/", "TestSet/", "Val/"]:
    for subset in ["covid_pneu_datasets", "normal_pneu_datasets", "healthy_datasets"]:
        get_slice_dataset(
                root_dir="/rdfs/fast/home/sunyingge/data/COV_19/0508/"+ split + subset,
                sub_dir=split + subset,
                to_dir="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/",
                phase="others",
                link="-",
                pro_fn=None,
                isprint=False
                )