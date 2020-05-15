import argparse, os, sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os.path import join as pj

from chenfeng.data_tools import BaseDataset, paths_for_dataset
from ASEUNet import SEResUNet

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--train_dir")
    parser.add_argument("-o", "--output_dir")
    # return parser.parse_args()
    # So that this works with jupyter
    return parser.parse_args(args=[
        "--train_dir",
        "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train/",
        "--output_dir",
        "/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/"
    ])

args = parse_args()

class Args(object):
    def __init__(self):
        self.gpu = "3"
        self.n_channels = 2
        self.n_classes = 1
        self.epochs = 80
        self.initial_epoch = 0
        self.global_step = 0
        self.batchsize = 4
        self.lr = 0.0005
        self.percentage = 0.1
        self.load = False
        self.save_cp = True
        self.img_size = (256, 256)
        self.description = "HRNET_KERAS_DEFAULT"

        # REMEMBER TO CHANGE THIS ACCORDINGLY!
        self.steps = [100000, 200000]
        self.lr = [1e-3, 5e-4, 25e-5]

model_args = Args()
os.environ["CUDA_VISIBLE_DEVICES"] = model_args.gpu

print("==>>Training set: ")
train_all, train_pos, train_neg = paths_for_dataset(args.train_dir,
    flags=["train"],
    seed=999,
    isprint=True)

train_paths_cons  = train_pos["train"]
train_paths_chan  = []
np.random.seed(999)
train_paths_cons = np.random.permutation(train_paths_cons).tolist()
train_paths_chan = np.random.permutation(train_paths_chan).tolist()

print("++"*30)
print(f"Length of train_paths_cons: {len(train_paths_cons)}")
print(f"Length of train_paths_chan: {len(train_paths_chan)}")

train_dataset = BaseDataset(train_paths_cons, train_paths_chan, img_size=model_args.img_size, choice="all",
                          image_key="image", mask_key="mask")
print(f"train_dataset: {len(train_dataset)}")

input_im = tf.placeholder(tf.float32, shape=(None, model_args.img_size[0], model_args.img_size[1], 1))
input_ann = tf.placeholder(tf.float32, shape=(None, model_args.img_size[0], model_args.img_size[1], 1))
pred = SEResUNet(input_im, num_classes=1, reduction=8, name_scope="SEResUNet")
ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_ann, logits=pred)
loss = tf.reduce_mean(ce)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.piecewise_constant(global_step, model_args.steps, model_args.lr)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(model_args.epochs):
        print(f"Epoch {epoch + 1}\n")
        num_batches = len(train_dataset) // model_args.batchsize
        for i in range(num_batches):
        # for i in range(10):
            print(f"Epoch progress: {i + 1} / {num_batches}")
            data_list = []
            for j in range(i * model_args.batchsize, (i + 1) * model_args.batchsize):
                data_list.append(train_dataset[j])
            data_ar = np.array(data_list)
            im_ar, ann_ar = data_ar[:,0,:,:,:], data_ar[:,1,:,:,:]

            # ret_loss, _ = sess.run([loss, optimizer],
            ret_loss, ret_pred, ret_ce, _ = sess.run([loss, pred, ce, optimizer],
                feed_dict={
                    input_im: im_ar,
                    input_ann: ann_ar,
                    })
            print(ret_loss)
        saver.save(sess, pj(args.output_dir, f"epoch_{epoch + 1}.ckpt"))
    

