import os, sys
from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf

sys.path.insert(0, "../../..")
from fast.ASEUNet_v2 import SEResUNet

class test(tf.keras.Model):
    def __init__(self, name="Model"):
        super(test, self).__init__(name=name)
        self.SEResUNet = partial(SEResUNet, num_classes=3, reduction=8, name_scope=name)

    def call(self, inputs):
        logits = self.SEResUNet(inputs)
        return logits

model = test()
model.compile(optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])
test_im = np.random.uniform(size=(1, 128, 128, 3))
pred = model.predict(test_im)
print(pred.shape)