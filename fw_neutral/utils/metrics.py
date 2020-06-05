import os, sys
import pickle, skimage.measure
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
from os.path import join as pj

if __name__ == "__main__":
    from data_proc import pneu_type
else:
    from .data_proc import pneu_type

class Patient():
    def __init__(self, thickness, tags):
        self.thickness = thickness
        self.lesion_info = [] # list of dict("size","slice_ind")
        self.tags = defaultdict(bool)
        for tag in tags:
            self.tags[tag] = True

def join_tags(tags):
    return ','.join(tags)

def check_tags(patient, tags):
    for tag in tags:
        if not patient.tags[tag]:
            return False
    return True

# need something interactive for this to work well
class Evaluation():
    def __init__(self, thickness_thres):
        self.dataset_stats = []
        self.thickness_threshold = thickness_thres
        # self.size_thres = [26300, 235000]
        # subset
        # self.subsets = {
        #     "thick": [],
        #     "thin": [],
        #     "common_pneu": [],
        #     "covid_pneu": []
        # }
        self.subsets = defaultdict(list)
        self.iou_thres = .5
        self.performance = {
            "covid_pneu": {"TP": 0, "FP": 0, "num_gt": 0, "num_pred": 0},
            "common_pneu": {"TP": 0, "FP": 0, "num_gt": 0, "num_pred": 0},
        }
    
    def subset_areas(self, tags):
        areas = []
        for p in self.subsets[join_tags(tags)]:
            for lesion in p.lesion_info:
                areas.append(lesion["area"])
        return areas

    def single_patient_stats(self, anno_file):
        """
            args:
                pred: 
        """
        anno_sitk = sitk.ReadImage(anno_file, sitk.sitkInt16)
        anno_ar = sitk.GetArrayFromImage(anno_sitk) # anno_ar: [C, H, W]
        patient_tags = []
        pneumonia_type = pneu_type(anno_file)
        patient_tags.append(pneumonia_type)
        assert pneumonia_type in ["common_pneu", "covid_pneu"], "Unknown type!"
        patient_tags.append("thick" if anno_sitk.GetSpacing()[-1] >= self.thickness_threshold else "thin")
        single_patient = Patient(anno_sitk.GetSpacing()[-1], patient_tags)
        for i in range(anno_ar.shape[0]):
            anno_ins, num_ins = skimage.measure.label(anno_ar[i,:,:], return_num=True, connectivity=2)
            for j in range(num_ins):
                instance_mask = anno_ins == j + 1
                single_patient.lesion_info.append(
                    dict(area=np.sum(instance_mask), slice_ind=i))
        self.dataset_stats.append(single_patient)
    
    def set_stats(self, pkl_file, output_dir, hist_range, subsets=[["common_pneu"], ["covid_pneu"]]):
        # subset_tags should be a list of lists of tags
        # instead of a list of single tags
        if len(self.dataset_stats):
            pickle.dump(self.dataset_stats, open(pj(output_dir, "set_stats.pkl"), "bw"))
        elif pkl_file != None:
            self.dataset_stats = pickle.load(open(pkl_file, "br"))
        for patient in self.dataset_stats:
            for tags in subsets:
                if check_tags(patient, tags):
                    # overload this
                    self.subsets[join_tags(tags)].append(patient)
        for tags in subsets:
            hist, bin_edges = np.histogram(self.subset_areas(tags), 
                bins=[0, 1000, 5000, 40000])
            print(f"\n{tags}")
            for i in range(len(hist)):
                print(f"[{bin_edges[i]}, {bin_edges[i + 1]}): {hist[i]}, {hist[i] / np.sum(hist):.4f}")
        # hist, bin_edges = np.histogram(self.dataset_stats, range=hist_range)
        # for i in range(len(hist)):
        #     print(f"[{bin_edges[i]}, {bin_edges[i + 1]}): {hist[i]}")

    def eval_single_patient(self, anno_file, pred):
        """
            args:
                pred: [C, H, W]
        """
        anno_sitk = sitk.ReadImage(anno_file, sitk.sitkInt16)
        anno_ar = sitk.GetArrayFromImage(anno_sitk) # anno_ar: [C, H, W]
        patient_tags = []
        pneumonia_type = pneu_type(anno_file)
        patient_tags.append(pneumonia_type)
        assert pneumonia_type in ["common_pneu", "covid_pneu"], "Unknown type!"
        patient_tags.append("thick" if anno_sitk.GetSpacing()[-1] >= self.thickness_threshold else "thin")
        single_patient = Patient(anno_sitk.GetSpacing()[-1], patient_tags)
        for i in range(anno_ar.shape[0]):
            gt_labels, gt_num = skimage.measure.label(anno_ar[i,:,:], return_num=True, connectivity=2)
            pred_labels, pred_num = skimage.measure.label(pred[i,:,:], return_num=True, connectivity=2)
            TP, FP = 0, 0
            # for each gt 
            # if iou>.5, TP+=1
            # else FN+=1
            # for each pred
            # if iou<.5, FP+=1
            # no TN?
            gt_matched_pred = [[] for _ in range(gt_num)]
            gt_labels_repeated = np.repeat(np.expand_dims(gt_labels, 0), gt_num, 0) # num_of_gt_instances, H, W
            gt_instance_mask = gt_labels_repeated == np.arange(start=1, stop=gt_num + 1).reshape([-1, 1, 1]) # num_of_gt_instances, H, W
            pred_labels_repeated = np.repeat(np.expand_dims(pred_labels, 0), gt_num, 0) # num_of_gt_instances, H, W
            itsect = pred_labels_repeated * gt_instance_mask

            for j in range(gt_num):
                itsect_area = np.sum(itsect[j,:,:] > 0)
                union = np.sum(gt_instance_mask[j,:,:]) - itsect_area
                # print(np.sum(gt_instance_mask[j,:,:]))
                for k in range(pred_num):
                    # When using '==' op pay attention that whether the index is 0 or 1 based
                    if (itsect[j,:,:] == k + 1).any(): 
                        # gt_matched_pred[j].append(k)
                        union += np.sum(pred_labels == k + 1)
                iou = float(itsect_area) / union
                if iou >= self.iou_thres:
                    TP += 1
                # else:
                #     self.FN += 1
            
            pred_labels_repeated = np.repeat(np.expand_dims(pred_labels, 0), pred_num, 0)
            pred_instance_mask = pred_labels_repeated == np.arange(start=1, stop=pred_num + 1).reshape([-1,1,1])
            gt_labels_repeated = np.repeat(np.expand_dims(gt_labels, 0), pred_num, 0)
            itsect = gt_labels_repeated * pred_instance_mask

            for j in range(pred_num):
                itsect_area = np.sum(itsect[j,:,:] > 0)
                union = np.sum(pred_instance_mask[j,:,:]) - itsect_area
                for k in range(gt_num):
                    if (itsect[j,:,:] == k + 1).any(): 
                        # gt_matched_pred[j].append(k)
                        union += np.sum(gt_labels == k + 1)
                iou = float(itsect_area) / union
                if iou < self.iou_thres:
                    FP += 1
            res_dict = self.performance[pneumonia_type]
            res_dict["TP"] += TP
            res_dict["FP"] += FP
            res_dict["num_gt"] += gt_num
            res_dict["num_pred"] += pred_num
            
        self.dataset_stats.append(single_patient)

    def lesion_level_performance(self):
        for key in self.performance:
            print(f"\n{key}:")
            rc = self.performance[key]["TP"] / self.performance[key]["num_gt"]
            pr = (self.performance[key]["num_pred"] - self.performance[key]["FP"]) / self.performance[key]["num_pred"]
            print(f"Recall: {rc:.4f}")
            print(f"Precision: {pr:.4f}")

# store a couple res for debugging     
if __name__ == "__main__":
    from os.path import join as pj
    import SimpleITK as sitk

    sys.path.insert(0, "../..")
    from fast.cf_mod.misc.utils import get_infos

    evaluation = Evaluation(3.0)
    # input_dir = "/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_0528_1357_35/eval_debug/covid_pneu_datasets/da1be23026fe174be22445a33064a85f"
    input_dir = "/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_0528_1357_35/eval_debug/"
    # im_dir = pj(input_dir, "da1be23026fe174be22445a33064a85f.nii.gz")
    # gt_dir = pj(input_dir, "da1be23026fe174be22445a33064a85f_pred.nii.gz")
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith("_pred.nii.gz"):
                pred_file = pj(root, f)
                gt_file = pred_file.replace("_pred.nii.gz", "_000-label.nii.gz")
                # im_file = pred_file.replace("_pred.nii.gz", ".nii.gz")
                assert os.path.exists(pred_file)                
                assert os.path.exists(gt_file)
                # assert os.path.exists(im_file)
                pred_ar = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
                evaluation.eval_single_patient(gt_file, pred_ar)
                print("hi")
                break
    evaluation.lesion_level_performance()
