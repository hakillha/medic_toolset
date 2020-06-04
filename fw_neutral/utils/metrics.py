import os
import pickle, skimage.measure
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
from os.path import join as pj

class Patient():
    def __init__(self, thickness, pneu_type, tags):
        self.thickness = thickness
        # self.pneu_type = pneu_type
        self.lesion_info = []
        self.tags = defaultdict(bool)
        for tag in tags:
            self.tags[tag] = True

# need something interactive for this to work well
class Evaluation():
    def __init__(self, args):
        self.dataset_stats = []
        self.thickness_threshold = args.thickness_thres
        # self.size_thres = [26300, 235000]
        # subset
        self.subsets = {
            "thick": [],
            "thin": [],
            "common_pneu": [],
            "covid_pneu": []
        }

    def eval_single_patient(self, anno_file):
        """
            args:
        """
        anno_sitk = sitk.ReadImage(anno_file, sitk.sitkInt16)
        anno_ar = sitk.GetArrayFromImage(anno_sitk) # anno_ar: [C, H, W]
        patient_tag = []
        # change the following to a function
        if "normal" in anno_file:
            patient_tag.append("common_pneu")
        elif "covid" in anno_file:
            patient_tag.append("covid_pneu")
        else:
            raise Exception("Pneumonia type error!")
        patient_tag.append("thick" if anno_sitk.GetSpacing()[-1] >= self.thickness_threshold else "thin")
        single_patient = Patient(anno_sitk.GetSpacing()[-1], patient_tag)
        for i in range(anno_ar.shape[0]):
            anno_ins, num_ins = skimage.measure.label(anno_ar[i,:,:], return_num=True, connectivity=2)
            for j in range(num_ins):
                instance_mask = anno_ins == j
                single_patient.lesion_info.append(
                    dict(size=np.sum(instance_mask),
                        slice_ind=i))
        self.dataset_stats.append(single_patient)

    def set_stats(self, pkl_file, output_dir, hist_range, subset_tags=["common_pneu", "covid_pneu"]):
        if os.path.exists(pkl_file):
            self.dataset_stats = pickle.load(open(pkl_file, "br"))
        elif len(self.dataset_stats):
            pickle.dump(self.dataset_stats, open(pj(output_dir, "set_stats.pkl"), "bw"))
        for patient in self.dataset_stats:
            if patient.pneu_type == "common_pneu":
                self.subsets[].append(patient)
            elif patient.pneu_type == "covid_pneu":
                self.subsets.append(patient)
        for tag in subset_tags:
            hist, bin_edges = np.histogram(self.subsets[tag])
            print(f"\n{tag}")
            for i in range(len(hist)):
                print(f"[{bin_edges[i]}, {bin_edges[i + 1]}): {hist[i]}")
        # hist, bin_edges = np.histogram(self.dataset_stats, range=hist_range)
        # for i in range(len(hist)):
        #     print(f"[{bin_edges[i]}, {bin_edges[i + 1]}): {hist[i]}")
        