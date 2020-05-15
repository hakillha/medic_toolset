# _*_ coding:utf-8 _*_
#!/usr/local/bin/python3 

from importlib import import_module
import os, glob, sys, shutil, re
import time, datetime
from tqdm import tqdm, trange, tgrange
import SimpleITK as sitk
import numpy as np
import cv2
from collections import defaultdict, Counter, Iterable

#import seaborn as sb
import matplotlib.pyplot as plt

import csv, json
#import xlrd, xlwt
import pickle

import subprocess as sp
import threading
import multiprocessing as mp
from functools import partial

from tensorflow import keras
import keras.backend as K
import tensorflow as tf

localtime = time.strftime("%Y-%m-%d", time.localtime())
heads = ["image_file", "label_file", "thickness"]
EPS  = 1.e-8
def get_abspath(path):
    path_ = os.path.expanduser(path)
    return os.path.abspath(path_)

_CONFIG = {
     "lobe_default"                     :       get_abspath("~/Production/GraphDef/LungLobe/HRNET_KERAS_2CH/default/model.graphdef"),
    
     "pneu_default"                     :       get_abspath("~/Production/GraphDef/LungPneumonia/HRNET_KERAS_2CH/default/model.graphdef"),
}

_TENSOR_NAME = {
    "lobe_default"                      :       {"inputs":"input_1:0", "outputs":"conv2d_86/truediv:0"},
    
    "pneu_default"                      :       {"inputs":"input_1:0", "outputs":"conv2d_86/Sigmoid:0"},
    
}



def read_excel(excel_path, sheet_num=0, col_inx=1, need_head=False):
    sheet_results = []
    start = 0 if need_head else 1
    xl_obj = xlrd.open_workbook(excel_path)
    sheet_name = xl_obj.sheet_names()[sheet_num]
    sheet = xl_obj.sheet_by_name(sheet_name)
    assert max(col_inx) < sheet.ncols, "Please Assure The Index Of Column Limited"
    for inx in range(start, sheet.nrows):
        raw = []
        col_list = col_inx if isinstance(col_inx, Iterabel) else range(col_inx) 
        for ii in col_list: 
            try:
                value = sheet_all.cell(inx, ii).value
                raw.append(value)
            except Exception as err:
                print(">> sheet: {}, raws: {}, columns: {}".format(sheet_num, inx, ii))
                print("|--", err)
                raw.append("")
        sheet_results.append(raw)
    return sheet_results 

def read_txt(txt):
    with open(txt, 'r') as f_txt_file:
        list_txt = f_txt_file.readlines()
        list_txt = [i.strip() for i in list_txt if len(i.strip()) > 0]
    return list_txt

def write_txt(path_list, record_file):
    with open(record_file, 'w') as f:
        for path in path_list:
            f.write(path+"\n")

def write_csv(all_infos, record_csv, heads=heads, need_head=True, mode="w"):
    with open(record_csv, mode, newline="") as f:
        writer = csv.writer(f, dialect="excel")
        if need_head:
            writer.writerow(heads)
        writer.writerows(all_infos)

def read_csv(record_csv,skip_head=True, isprint=False):
    all_paths = []
    with open(record_csv, 'r') as f:
        reader = csv.reader(f)
        if skip_head:
            file_heads = next(reader)
        for ii in reader:
            all_paths.append(ii)
        if isprint:
            print("Get Lines: ", len(all_paths))
        if skip_head:
            return all_paths, file_heads
        else:
            return all_paths, []
        
def get_most(list_dict):
    count_dict = {}
    for key, value in list_dict.items():
        result = Counter(value).most_common(1)[0][0] 
        count_dict[key] = result
    return count_dict

def get_unique(info_paths):
    unique  = set()
    for infos in tqdm(info_paths):
        _, lab_file = infos[:2]
        lab_ori = sitk.ReadImage(lab_file)
        lab_arr = sitk.GetArrayFromImage(lab_ori)
        uni_num = np.unique(lab_arr)
        unique.update(set(uni_num))
    return sorted(list(unique))

def copy_dirs(dir_path, flags=["", ""]):
    to_dir = dir_path.replace(*flags) 
    if not os.path.exists(to_dir):
        shutil.copytree(dir_path, to_dir)
        

def get_chest_mask(img, threshold=-300):
    assert img.ndim==3 , "3D Array Needed"
    img_arr = np.asarray(img > threshold, dtype=np.uint8)
    chest_mask = np.zeros_like(img_arr, dtype=np.uint8)
    for inx in range(img_arr.shape[0]):
        contours,  hierarchy= cv2.findContours(img_arr[inx, ...], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area_max = 0 ; cnt_want = None
        if len(contours):
            for cnt in contours:
                area = cv2.contourArea(cnt, False)
                if area > area_max: area_max = area; cnt_want = cnt 
            cv2.fillPoly(chest_mask[inx], [cnt_want], 1) 
    return chest_mask

def letterbox_slice(sli, padding=0., shape=(256, 256), interpolation=cv2.INTER_LINEAR):
    ih, iw = sli.shape
    h, w = shape
    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    sli = cv2.resize(sli, (nw, nh), interpolation=interpolation)
    sli_ = np.ones(shape, dtype=np.float32)*padding
    pd_h = int((h-nh)/2.); pd_w = int((w-nw)/2.)
    sli_[pd_h:pd_h+nh, pd_w:pd_w+nw] = sli
    return sli_
def letterbox_box(arr, padding=0, shape=(256, 256), interpolation=cv2.INTER_LINEAR): 
    pass


def resize3d(img, shape=(256, 256), interpolation=cv2.INTER_NEAREST):
    #assert img.ndim==3 , "3D Array Needed"
    img_list = []
    for inx in range(img.shape[0]):
        sli = cv2.resize(img[inx,  ...], shape[::-1], interpolation=interpolation)
        img_list.append(sli)
    return np.asarray(img_list)

def get_infos(root_dir, link="-",  localtime=localtime, isprint=True):
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
                            get_paths.append([img_file, lab_file, localtime])
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
                    get_paths.append([img_file, lab_file, localtime]) 
    if isprint: print("all_picked: {}".format(get_num))
    return get_paths 

def get_infos_single(root_dir, localtime=localtime, isprint=True):
    get_paths = []
    get_num = 0
    for root, dirs, files in os.walk(root_dir, topdown=True):
        if len(files):
            for inx in files:
                if inx.endswith(".nii.gz"):
                    img_file = os.path.join(root, inx)
                    get_num += 1
                    get_paths.append([img_file, "", localtime])


    if isprint: print("all_picked single: {}".format(get_num))
    return get_paths

def get_infos_multi(root_dir, 
                    shuffix=[".nii.gz", "-label.nii.gz", "_ant-label.nii.gz"],
                    isprint=True,
                   ):
    r"""
    shuffix: assign of the element in shuffix_list should be longer and longer
    """
    mask = True
    shuffix = sorted(shuffix, key=lambda x:len(x), reverse=True)
    length = len(shuffix)
    for inx in shuffix:
        if len(inx)==0: mask = False
    assert mask, "The element in shuffix should not be the zero-length"
    info_paths = []
    for root, dirs, files in os.walk(root_dir):
        if len(files) == len(shuffix):
            infos = ["" for _ in shuffix]
            for inx in files:
                for num, ind in enumerate(shuffix):
                    if inx.endswith(ind):
                        infos[length-1-num] = os.path.join(root, inx); break
            info_paths.append(infos)
    if isprint: print(f"picked: {len(info_paths)}")
    return info_paths    
## model to graphdef   
def tf_to_graphdef(meta_file, ckpt_file, to_dir, config, output_names, graphdef_name='model.graphdef'):
    if not os.path.exists(to_dir): os.makedirs(to_dir)
    with tf.Graph().as_default() as crop_graph:
        saver = tf.train.import_meta_graph(meta_file)
        with tf.Session(config=config).as_default() as crop_sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            print('convert tensorflow_model into graphdef...')
            ckpt = ckpt_file 
            saver.restore(crop_sess, ckpt)
            graph_def = crop_graph.as_graph_def()
            node_list = [node for node in graph_def.node]
            op_list = [op for op in tf.get_default_graph().get_operations()]
            for op in op_list:
                # if op.type == "Input":
                #     print(op.inputs)
                # if "conv_final" in op.name:
                #     print(op.outputs)
                if "SEResUNet/Input/Const" in op.name:
                    print(op.name)
                    # print(op.inputs.name)
                    print(op.outputs)
            # for node in node_list:
            #     # if "SEResUNet/conv_final/" in node.name or "Placeholder" in node.name:
            #     if "sigmoid" in node.name or "Sigmoid" in node.name or "entropy" in node.name:
            #         print(node.name)
            backlist = [node.name for node in graph_def.node if 'IsVariableInitialized' in node.name]
            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    node.op = 'Add'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'Assign':
                    node.op = 'Identity'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                    if 'validate_shape' in node.attr: del node.attr['validate_shape']
                    if len(node.input) == 2:
                        node.input[0] = node.input[1]
                        del node.input[1]
            for node in graph_def.node:
                node.device = ""
                #print(node.name)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                crop_sess,
                graph_def,
                output_names,
                variable_names_blacklist = backlist
            )
            tf.train.write_graph(output_graph_def, to_dir, graphdef_name, as_text=False)
            # tf.train.write_graph(graph_def, to_dir, graphdef_name, as_text=False)
            print(">>>done<<<")

def keras_to_graphdef(network, weights_file, to_dir, config, graphdef_name='model.graphdef', **args):
    with tf.Graph().as_default() as crop_graph:
        crop_model = network(**args)
        with tf.Session(config=config).as_default() as crop_sess:
            K.set_learning_phase(0)
            crop_model.load_weights(weights_file)
            print('convert keras_model into graphdef...')
            output_names = [crop_model.output.name.split(':')[0]]
            graph_def = crop_graph.as_graph_def()
            backlist = [node.name for node in graph_def.node if 'IsVariableInitialized' in node.name]
            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    node.op = 'Add'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'Assign':
                    node.op = 'Identity'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                    if 'validate_shape' in node.attr: del node.attr['validate_shape']
                    if len(node.input) == 2:
                        node.input[0] = node.input[1]
                        del node.input[1]
            for node in graph_def.node:
                node.device = ""
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                crop_sess,
                graph_def,
                output_names,
                variable_names_blacklist = backlist
            )
            if not os.path.exists(to_dir): os.makedirs(to_dir)
            tf.train.write_graph(output_graph_def, to_dir, graphdef_name, as_text=False)
            print(">>>done<<<")
## model from graph
def graph_from_graphdef(net, config, CONFIG=_CONFIG, TENSOR_NAME=_TENSOR_NAME):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(CONFIG[net], "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            inputs = graph.get_tensor_by_name(TENSOR_NAME[net]["inputs"])
            outputs = graph.get_tensor_by_name(TENSOR_NAME[net]["outputs"])
    sess = tf.Session(graph=graph, config=config)
    return [graph, sess], [inputs, outputs]

def visualize(unit=4, cmap="gray", **kwargs):
    amount  = len(kwargs)
    from math import sqrt
    h = w = int(sqrt(amount))+1
    fig = plt.figure(figsize=(h*unit, w*unit))
    for inx, (name, img) in enumerate(kwargs.items()):
        plt.subplot(h, w, inx+1)
        sub = plt.imshow(img, cmap=cmap)
        plt.colorbar(sub)
        plt.title(name)
    plt.show()

def pre_lab_arr(arr, description="",  mode="disease", dtype=None):
    assert mode.lower() in ["disease", "lobe"], "Please Specify The Right Mode"
    assert isinstance(arr, np.ndarray), "Only Available DataType For '<class np.ndarray>'"
    assert (dtype is None) or isinstance(dtype, type), "Please Specify The Right DataType"
    lab_arr = arr.copy()
    if mode.lower() == "lobe":
        if "label" in description.lower():
            pos = ((lab_arr>0)*(lab_arr<=5)) + ((lab_arr<=224)*(lab_arr>=219))
            lab_arr[~pos] = 0
            lab_arr[lab_arr==219] = 1
            lab_arr[lab_arr==220] = 2
            lab_arr[lab_arr==221] = 3
            lab_arr[lab_arr==222] = 4
            lab_arr[lab_arr==223] = 5
    if mode.lower() == "disease":
        disease_phase = "covid"
        if "covid_pneu" in description.lower():
            disease_phase = "covid"
        elif "normal_pneu" in description.lower() and ("/301" in description.lower() or "/302" in description.lower()):
            disease_phase = "301"
        elif "normal_pneu" in description.lower() and "/xiamen/" in description.lower():
            disease_phase = "xiamen"
        if disease_phase == "301":
            lab_arr[lab_arr != 71] = 0
            lab_arr[lab_arr == 71] = 1
        elif disease_phase == "covid":
            lab_arr[(lab_arr>194)*(lab_arr<300)] = 71
            lab_arr[lab_arr!=71] = 0
            lab_arr[lab_arr==71] = 1
        elif disease_phase == "xiamen":
            lab_arr[lab_arr==1] = 0
            lab_arr[(lab_arr<78)*(lab_arr>71)] = 1
            lab_arr[lab_arr != 1] = 0
        else:
            lab_arr[lab_arr >0.5] = 1
        
    if dtype is None:
        
        return np.expand_dims(lab_arr, axis=-1)
    else:
        return np.expand_dims(lab_arr.astype(dtype), axis=-1)
    
def pre_img_arr_2ch(arr, 
                    min_ch1=-200., max_ch1=500.,
                    min_ch2=-1000.,max_ch2=-100.,
                   ):
    img_arr = arr.copy()
    arr_ch1 = np.clip(img_arr, min_ch1, max_ch1)
    arr_min = np.amin(arr_ch1); arr_max = np.amax(arr_ch1)
    arr_ch1 = (arr_ch1 - arr_min + EPS)/(arr_max - arr_min + EPS)
    
    arr_ch2 = np.clip(img_arr, min_ch2, max_ch2)
    arr_min = np.amin(arr_ch2); arr_max = np.amax(arr_ch2)
    arr_ch2 = (arr_ch2 - arr_min + EPS)/(arr_max - arr_min + EPS)
    img_arr = np.stack([arr_ch1, arr_ch2], axis=-1)
    return img_arr

def pre_img_arr_1ch(arr, 
                    min_ch1=-1400., max_ch1=800.,
                ):
    img_arr = arr.copy()
    arr_ch1 = np.clip(img_arr, min_ch1, max_ch1)
    arr_min = np.amin(arr_ch1); arr_max = np.amax(arr_ch1)
    arr_ch1 = (arr_ch1 - arr_min + EPS)/(arr_max - arr_min + EPS)
    
    img_arr = np.expand_dims(arr_ch1, axis=-1)
    return img_arr

def pkl2dict(pkl_file, score_fn=None):
    def _score2level(score):
        if score >= 0.7: 
            return str(0)
        elif score >= 0.5:
            return str(1)
        elif score >= 0.3:
            return str(2)
        elif score >=0.2:
            return str(3)
        else:
            return str(4)
    if not score_fn:
        score_fn = _score2level
    level_dict = defaultdict(list)
    all_result = pickle.load(
        open(pkl_file, "rb")
        )
    for ret in all_result:
        file, score, _ = ret
        level = score_fn(score)
        level_dict[level].append(os.path.dirname(file))
    return level_dict

def pkl2dict_score(pkl_file, score_fn=None):
    def _score2level(score):
        if score >= 0.7: 
            return str(0)
        elif score >= 0.5:
            return str(1)
        elif score >= 0.3:
            return str(2)
        elif score >=0.2:
            return str(3)
        else:
            return str(4)
    if not score_fn:
        score_fn = _score2level
    level_dict = defaultdict(list)
    all_result = pickle.load(
        open(pkl_file, "rb")
        )
    for ret in all_result:
        file, score, _ = ret
        level = score_fn(score)
        level_dict[level].append(score)
    return level_dict

 
