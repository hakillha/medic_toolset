import argparse, os, sys
import pickle, skimage.measure
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
from os.path import join as pj

if __name__ == "__main__":
    from data_proc import Pneu_type
else:
    from .data_proc import Pneu_type

class Patient():
    def __init__(self, fname, thickness, tags, pneu_type):
        self.fname = fname
        self.thickness = thickness
        self.lesion_info = [] # list of Lesion() instances
        self.pred_info = []
        self.pneu_type = pneu_type
        self.tags = defaultdict(bool)
        for tag in tags:
            self.tags[tag] = True

class SegInstance():
    def __init__(self, area, area_interval, slice_ind, iou=None):
        self.area = area
        for k, v in area_interval.items():
            if area >= v[0] and area < v[1]:
                self.size = k
        self.slice_ind = slice_ind
        self.iou = iou

class Lesion(SegInstance):
    def __init__(self, detected, *args):
        super(Lesion, self).__init__(*args)
        self.detected = detected

class PredInstance(SegInstance):
    def __init__(self, matched, *args):
        super(PredInstance, self).__init__(*args)
        self.matched = matched

def join_tags(tags):
    # tags = sorted(tags)
    return ','.join(tags)

def check_tags(patient, tags):
    if not isinstance(tags, list):
        tags = [tags]
    if tags == ["all"]:
        return True
    for tag in tags:
        if not patient.tags[tag]:
            return False
    return True

# need something interactive for this to work well
class Evaluation():
    def __init__(self, thickness_thres=3.0):
        self.dataset_stats = []
        self.thickness_threshold = thickness_thres
        # self.size_thres = [26300, 235000]
        self.subsets = defaultdict(list)
        # subset example
        # self.subsets = {
        #     "thick": [],
        #     "thin": [],
        #     "common_pneu": [],
        #     "covid_pneu": []
        # }
        self.iou_thres = .5
        self.area_interval = {
            "small": [0, 1000],
            "middle": [1000, 5000],
            "large": [5000, 1e10],
            "all": [0, 1e10]
        }
        self.class_map = {
            "covid_pneu": 1,
            "common_pneu": 2,
        }

    def subset_areas(self, tags):
        areas = []
        for p in self.subsets[join_tags(tags)]:
            for lesion in p.lesion_info:
                areas.append(lesion.area)
        return areas

    def single_patient_stats(self, anno_file):
        """
            args:
                pred: 
        """
        anno_sitk = sitk.ReadImage(anno_file, sitk.sitkInt16)
        anno_ar = sitk.GetArrayFromImage(anno_sitk) # anno_ar: [C, H, W]
        patient_tags = []
        pneumonia_type = Pneu_type(anno_file)
        patient_tags.append(pneumonia_type)
        assert pneumonia_type in ["common_pneu", "covid_pneu"], "Unknown type!"
        patient_tags.append("thick" if anno_sitk.GetSpacing()[-1] >= self.thickness_threshold else "thin")
        single_patient = Patient(anno_sitk.GetSpacing()[-1], patient_tags)
        for i in range(anno_ar.shape[0]):
            anno_ins, num_ins = skimage.measure.label(anno_ar[i,:,:], return_num=True, connectivity=2)
            for j in range(num_ins):
                instance_mask = anno_ins == j + 1
                single_patient.lesion_info.append(Lesion(np.sum(instance_mask), i))
        self.dataset_stats.append(single_patient)
    
    def set_stats(self, pkl_file, output_dir, hist_range, 
        subsets=[["common_pneu"], ["covid_pneu"], ["all"]]):
        # subset_tags should be a list of lists of tags
        # instead of a list of single tags
        if len(self.dataset_stats):
            pickle.dump(self.dataset_stats, open(pj(output_dir, "set_stats.pkl"), "bw"))
        elif pkl_file != None:
            self.dataset_stats = pickle.load(open(pkl_file, "br"))
        for patient in self.dataset_stats:
            for tags in subsets:
                if check_tags(patient, tags):
                    # overload join_tags
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
        pneumonia_type = Pneu_type(anno_file)
        patient_tags.append(pneumonia_type)
        assert pneumonia_type in ["common_pneu", "covid_pneu"], "Unknown type!"
        patient_tags.append("thick" if anno_sitk.GetSpacing()[-1] >= self.thickness_threshold else "thin")
        single_patient = Patient(anno_file, anno_sitk.GetSpacing()[-1], patient_tags, pneumonia_type)
        for i in range(anno_ar.shape[0]):
            gt_labels, gt_num = skimage.measure.label(anno_ar[i,:,:], return_num=True, connectivity=2)
            pred_labels, pred_num = skimage.measure.label(pred[i,:,:], return_num=True, connectivity=2)
            gt_labels_repeated = np.repeat(np.expand_dims(gt_labels, 0), gt_num, 0) # num_of_gt_instances, H, W
            gt_instance_mask = gt_labels_repeated == np.arange(start=1, stop=gt_num + 1).reshape([-1,1,1]) # num_of_gt_instances, H, W
            pred_labels_repeated = np.repeat(np.expand_dims(pred_labels, 0), gt_num, 0) # num_of_gt_instances, H, W
            itsect = pred_labels_repeated * gt_instance_mask
            for j in range(gt_num):
                itsect_area = np.sum(itsect[j,:,:] > 0)
                gt_area = np.sum(gt_instance_mask[j,:,:])
                union = gt_area - itsect_area
                for k in range(pred_num):
                    # When using '==' op pay attention that whether the index is 0 or 1 based
                    if (itsect[j,:,:] == k + 1).any(): 
                        union += np.sum(pred_labels == k + 1)
                iou = float(itsect_area) / union
                single_patient.lesion_info.append(
                    Lesion(iou > self.iou_thres, gt_area, self.area_interval, i, iou))
            
            pred_labels_repeated = np.repeat(np.expand_dims(pred_labels, 0), pred_num, 0)
            pred_instance_mask = pred_labels_repeated == np.arange(start=1, stop=pred_num + 1).reshape([-1,1,1])
            gt_labels_repeated = np.repeat(np.expand_dims(gt_labels, 0), pred_num, 0)
            itsect = gt_labels_repeated * pred_instance_mask
            for j in range(pred_num):
                itsect_area = np.sum(itsect[j,:,:] > 0)
                pred_area = np.sum(pred_instance_mask[j,:,:])
                union = pred_area - itsect_area
                for k in range(gt_num):
                    if (itsect[j,:,:] == k + 1).any(): 
                        union += np.sum(gt_labels == k + 1)
                iou = float(itsect_area) / union
                single_patient.pred_info.append(
                    PredInstance(iou > self.iou_thres, pred_area, self.area_interval, i, iou))
        self.dataset_stats.append(single_patient)

    def lesion_level_performance(self):
        subsets = defaultdict(lambda : {"pred": [], "gt": []})
        for p in self.dataset_stats:
            for pneu in ["common_pneu", "covid_pneu"]:
                if check_tags(p, pneu):
                    for size in ["small", "middle", "large", "all"]:
                        for pred in p.pred_info:
                            if pred.size == size or size == "all":
                                subsets[join_tags([pneu, size])]["pred"].append(pred)
                        for gt in p.lesion_info:
                            if gt.size == size or size == "all":
                                subsets[join_tags([pneu, size])]["gt"].append(gt)
        
        for k, v in subsets.items():
            rc = len([gt for gt in v["gt"] if gt.detected]) / len(v["gt"])
            pr = len([pred for pred in v["pred"] if pred.matched]) / len(v["pred"])
            print('\n' + k)
            print(f"Recall: {rc:.4f}, Precision:{pr:.4f}")
    
    def gen_seginstance_entries(self, patient):
        gt_entries = [[gt.area, gt.iou, self.class_map[patient.pneu_type]] for gt in patient.lesion_info]
        pred_entries = [[pred.area, pred.iou, self.class_map[patient.pneu_type]] for pred in patient.pred_info]
        return gt_entries, pred_entries

    def gen_filters(self, cls_filter=None, area_filter=None):
        if cls_filter is None:
            cls_filter = self.class_map.keys()
        if area_filter is None:
            area_filter = self.area_interval.keys()
        filters = []
        for cls_k in cls_filter:
            for area_k in area_filter:
                filters.append((cls_k, self.class_map[cls_k], area_k, self.area_interval[area_k]))
        return filters

    def cal_pr_rc(self, in_array, class_v, area_v):
        in_pick = np.ones_like(in_array[:,0])
        in_pick = in_pick * (in_array[:,0] >= area_v[0]) * (in_array[:,0] < area_v[1])
        in_pick = in_pick * (in_array[:,2] == class_v)
        num_total = np.sum(in_pick)
        in_pick = in_pick * (in_array[:,1] >= self.iou_thres)
        return np.sum(in_pick) / num_total if num_total else None
        
    def ds_level_performance_np(self):
        gt_list, pred_list = [], []
        for p in self.dataset_stats:
            gt_entries, pred_entries = self.gen_seginstance_entries(p)
            gt_list += gt_entries
            pred_list += pred_entries
        gt_ar, pred_ar = np.array(gt_list), np.array(pred_list)
        for class_k, class_v, area_k, area_v in self.gen_filters():
            rc = self.cal_pr_rc(gt_ar, class_v, area_v)
            pr = self.cal_pr_rc(pred_ar, class_v, area_v)
            print(f"\n{class_k}, {area_k}:")
            print(f"Recall: {rc:.4f}, Precision:{pr:.4f}, F1-score: {2 * rc * pr / (rc + pr):.4f}")
    
    def patient_level_pf(self):
        for class_k, class_v, area_k, area_v in self.gen_filters():
            rcs, prs = [], []
            for p in self.dataset_stats:
                gt_entries, pred_entries = self.gen_seginstance_entries(p)
                rc = self.cal_pr_rc(np.array(gt_entries), class_v, area_v)
                pr = self.cal_pr_rc(np.array(pred_entries), class_v, area_v)
                if rc: rcs.append(rc)
                if pr: prs.append(pr)
            rc, pr = np.mean(rcs), np.mean(prs)
            print(f"\n{class_k}, {area_k}:")
            print(f"Recall: {rc:.4f}, Precision:{pr:.4f}, F1-score: {2 * rc * pr / (rc + pr):.4f}")

def pasrse_args():
    parser = argparse.ArgumentParser("""""")
    parser.add_argument("-o", "--output_dir", help="If the result already exists, it will be loaded instead.",
        default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_0528_1357_35/res/epoch_14.pkl")
    parser.add_argument("-i", "--input_dir",
        default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_0528_1357_35/eval_0604/"
        # default="/rdfs/fast/home/sunyingge/data/models/workdir_0522/SEResUNET_0528_1357_35/eval_debug/"
        )
    return parser.parse_args()

# store a couple res for debugging     
if __name__ == "__main__":
    from os.path import join as pj
    import SimpleITK as sitk

    args = pasrse_args()
    if os.path.exists(args.output_dir):
        evaluation = pickle.load(open(args.output_dir, "br"))
        # evaluation.lesion_level_performance()
        # evaluation.ds_level_performance_np()
        evaluation.patient_level_pf()
    else:
        evaluation = Evaluation()
        for root, dirs, files in os.walk(args.input_dir):
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
        pickle.dump(evaluation, open(args.output_dir, "bw"))
        # evaluation.lesion_level_performance()
        # evaluation.ds_level_performance_np()
        evaluation.patient_level_pf()