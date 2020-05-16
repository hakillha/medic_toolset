import argparse, os, sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os.path import join as pj

# When the TF module is large enough create a seperate folder for it
# TODO: Change working directory to point to folder of this script
sys.path.insert(0, "..") 
from fast.cf_mod.misc.data_tools import BaseDataset, paths_for_dataset
# TODO: unify the process of building models
from fast.ASEUNet import SEResUNet

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("mode", choices=["train", "only_eval"])
    parser.add_argument("gpu", help="Choose GPU.")
    
    # TODO: add an automatically generated (time-related) postfix
    parser.add_argument("-o", "--output_dir",
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0516/")

    # Training mode related
    parser.add_argument("--train_dir", help="Training set directory.",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train/",)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume", help="Checkpoint file to resume from.",
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/newf/epoch_11.ckpt")

    return parser.parse_args()
    # So that this works with jupyter
    # return parser.parse_args(args=[
    #     "--train_dir",
    #     "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train/",
    #     "--output_dir",
    #     "/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/"
    # ])

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

class Args(object):
    def __init__(self):
        self.batchsize = args.batch_size
        self.img_size = (256, 256)
        self.epochs = 40
        # self.n_channels = 2
        # self.n_classes = 1
        # self.initial_epoch = 0
        # self.global_step = 0
        # self.lr = 0.0005
        # self.percentage = 0.1
        # self.load = False
        # self.save_cp = True
        # self.description = "HRNET_KERAS_DEFAULT"

        self.steps = [4, 8] # The unit is epoch
        self.lr = [1e-3, 5e-4, 25e-5]
        self.max_to_keep = 30
training_args = Args()

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
train_dataset = BaseDataset(train_paths_cons, train_paths_chan, img_size=training_args.img_size, choice="all",
    image_key="image", mask_key="mask")
print(f"train_dataset: {len(train_dataset)}")

input_im = tf.placeholder(tf.float32, shape=(None, training_args.img_size[0], training_args.img_size[1], 1))
input_ann = tf.placeholder(tf.float32, shape=(None, training_args.img_size[0], training_args.img_size[1], 1))
pred = SEResUNet(input_im, num_classes=1, reduction=8, name_scope="SEResUNet")
ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_ann, logits=pred)
loss = tf.reduce_mean(ce)
global_step = tf.Variable(0, trainable=False)
num_batches = len(train_dataset) // training_args.batchsize
steps = [num_batches * epoch_step for epoch_step in training_args.steps]
print(f"Steps to drop LR: {steps}")
learning_rate = tf.train.piecewise_constant(global_step, steps, training_args.lr)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
saver = tf.train.Saver(max_to_keep=training_args.max_to_keep)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if args.resume:
        saver.restore(sess, args.resume)
    if args.resume:
        # This needs to be changed if the naming rule changes
        epoch = int((os.path.basename(args.resume).split('.')[0]).split('_')[-1])
    else:
        epoch = 0
    while epoch < training_args.epochs:
        print(f"Epoch {epoch + 1}\n")
        for i in range(num_batches):
        # for i in range(10):
            print(f"Epoch progress: {i + 1} / {num_batches}")
            data_list = []
            for j in range(i * training_args.batchsize, (i + 1) * training_args.batchsize):
                data_list.append(train_dataset[j])
            data_ar = np.array(data_list)
            im_ar, ann_ar = data_ar[:,0,:,:,:], data_ar[:,1,:,:,:]

            # ret_loss, _ = sess.run([loss, optimizer],
            ret_loss, ret_pred, ret_ce, _ = sess.run([loss, pred, ce, optimizer],
                feed_dict={input_im: im_ar, input_ann: ann_ar,})
            print(ret_loss)
        saver.save(sess, pj(args.output_dir, f"epoch_{epoch + 1}.ckpt"))
        epoch += 1
    
