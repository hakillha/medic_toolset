import json
from collections import namedtuple

class SafeDict(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        # For some reason this doesn't work in multiprocessing
        # if key not in self:
        #     print(self)
        #     raise KeyError(f"'{key}' is not a legal key.")
        dict.__setitem__(self, key, value)
    
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

default_lr_schedules = {
    "epoch_wise_constant": {
        "type": "epoch_wise_constant",
        "epoch_to_drop_lr": [8, 14],
        "lr": [1e-3, 5e-4, 25e-5]
    },
    "halved": {
        "type": "halved",
        "period": 2,
        "init_lr": 1e-3,
        "first_epoch2drop": 2,
        "decay_rate": 4
    },
    "cos_decay": {
        "type": "cos_decay",
        "epoch_period": 20,
        "lr": 1e-4
    }
}

# When we extend this to other frameworks we will need subclass this
class Config():
    def __init__(self):
        # default values
        self.batch_size = 32
        self.im_size = (256, 256) # w, h
        self.max_epoch = 40 # maximum number of epochs you can run
        self.num_class = 1
        self.trainset = SafeDict({
            "val_ratio": .2,
            # Ratio being 0 means not using any healthy samples. Set this to 0 < ratio < 1 when #neg samples is larger(???).
            # Or set to "all" if you want all the negative samples. Set this to smaller than 1 to have an opposite
            # behavior.
            "pn_ratio": 0, 
            # When provided, directly loads dirs from this file over other settings
            "data_list": "", 
            "patient_filter_file": "",
            "md5_map": "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/MD5_map.csv",
            # "slice_filter_file": "",
            "quality": []
        })
        self.valset = SafeDict({
            "datalist_dir": None
        })
        self.trainset_eval = SafeDict({
            "root_dir": None,
            "pos_or_neg": "all", 
            "patient_filter_file": "",
            "md5_map": "/rdfs/fast/home/sunyingge/data/COV_19/prced_0512/MD5_map.csv",
            "quality": []
        })
        self.testset = SafeDict({
            "include_healthy": False 
        })

        self.preprocess = SafeDict({
            "normalize": None,
            "cropping": False,
            "cropping_ct_thres": None,
            "cropping_train_randomness": None,
            "resize": True,
            "flip": False, # horizontal flip
        })

        self.network = SafeDict({
            "name": "SEResUNet", # "SEResUNet" or "UNet"
            "reconstruct": False, # true or false
            "weight_decay": False, # small float point number of false
            "norm_layer": "BN", # "GN" or "BN", this option only works for vanilla unet
        })
        self.optimizer = None
        self.lr_schedule = default_lr_schedules["epoch_wise_constant"]
        self.loss = None

        # The old default setting kept for reference
        # self.eval = {"ct_interval": [-1400, 800], "norm_by_interval": False}

    def load_from_json(self, cf_file):
        with open(cf_file) as f:
            cf_dict = json.load(f)
        # or use the give a default value dictionary indexing interface
        self.batch_size = cf_dict["batch_size"]
        self.im_size = tuple([dim for dim in cf_dict["im_size"]])
        self.max_epoch = cf_dict["max_epoch"]
        self.num_class = cf_dict["num_class"]
        
        # The following two are essential so are enforced
        for key in cf_dict["trainset"]:
            self.trainset[key] = cf_dict["trainset"][key]
        for key in cf_dict["valset"]:
            self.valset[key] = cf_dict["valset"][key]
        if "trainset_eval" in cf_dict:
            for key in cf_dict["trainset_eval"]:
                self.trainset_eval[key] = cf_dict["trainset_eval"][key]
        if "testset" in cf_dict:
            for key in cf_dict["testset"]:
                self.testset[key] = cf_dict["testset"][key]

        # this should be a list in the json file 
        # indicating the order of preprocessing?
        assert "normalize" in cf_dict["preprocess"] # This is explicitly required now
        for key in cf_dict["preprocess"]:
            self.preprocess[key] = cf_dict["preprocess"][key]

        for key in cf_dict["network"]:
            self.network[key] = cf_dict["network"][key]

        self.optimizer = cf_dict["optimizer"]
        assert "lr_schedule" in cf_dict
        self.lr_schedule = cf_dict["lr_schedule"]

        if "multiclass_loss" in cf_dict:
            self.loss = cf_dict["multiclass_loss"]
        else:
            self.loss = cf_dict["loss"]
        if self.num_class == 1:
            assert self.loss in ["sigmoid", "hbloss_dice_focal", "generalized_dice_loss", "dice_loss", "focal",
                "hbloss_dice_focal_v2", "hbloss_dice_ce", "hbloss_gendice_ce"]
        else:
            assert self.loss in ["softmax", "sigmoid"]
        # use the default setting which is the old setting
        # for the old cfg files where eval configuration is missing
        # new cfg files have to overwrite this to work correctly
        # if "eval" in cf_dict:
        #     self.eval = cf_dict["eval"]

class EvalConfig():
    def __init__(self):
        self.person_status = None
    
    def load_from_json(self, cf_file):
        with open(cf_file) as f:
            cf_dict = json.load(f)
        self.person_status = cf_dict["person_status"]