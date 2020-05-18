import argparse, os, sys, time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os.path import join as pj

# When the TF module is large enough create a seperate folder for it
# TODO: Change working directory to point to folder of this script
sys.path.insert(0, "..") 
# from fast.cf_mod.misc.data_tools import BaseDataset, paths_for_dataset
from fast.cf_mod.misc.data_tools import BaseDataset
from utils.data_proc import paths_from_data
# TODO: unify the process of building models
from fast.ASEUNet import SEResUNet

def parse_args():
    parser = argparse.ArgumentParser(
        """You need to be careful with resuming while also changing the batch size 
        with this script since it would change how many steps to take to drop LR.""")
    parser.add_argument("mode", choices=["train", "only_eval"])
    parser.add_argument("gpu", help="Choose GPU.")
    
    # TODO: add an automatically generated (time-related) postfix
    parser.add_argument("-o", "--output_dir",
        help="""You only need to provide a prefix which will be automatically be 
            complemented by time to keep things distincive easily.""",
        default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_")

    # Training mode related
    parser.add_argument("--train_dir", help="Training set directory.",
        default="/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/Train_0518/",)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume", help="Checkpoint file to resume from.",
        # default="/rdfs/fast/home/sunyingge/data/models/work_dir_0514/SEResUNET_0514/newf/epoch_11.ckpt")
    )

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
# _, train_pos, _ = paths_for_dataset(args.train_dir,
#     flags=["train"],
#     seed=999,
#     isprint=True)
train_paths  = paths_from_data(args.train_dir)
np.random.seed(999)
train_paths = np.random.permutation(train_paths).tolist()
print("++"*30)
print(f"Number of training samples: {len(train_paths)}")
train_dataset = BaseDataset(train_paths, [], img_size=training_args.img_size, choice="all",
    image_key="im", mask_key="mask")
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

output_dir = args.output_dir + time.strftime("%m_%d_%H_%M", time.localtime())
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
saver = tf.train.Saver(max_to_keep=training_args.max_to_keep)

num_para = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print("Total number of trainable parameters: {:.3}M.\n".format(float(num_para)/1e6))
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if args.resume:
        saver.restore(sess, args.resume)
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

            ret_loss, ret_pred, ret_ce, _ = sess.run([loss, pred, ce, optimizer],
                feed_dict={input_im: im_ar, input_ann: ann_ar,})
            print(ret_loss)
        saver.save(sess, pj(args.output_dir, f"epoch_{epoch + 1}.ckpt"))
        epoch += 1
    
