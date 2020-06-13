import json

# When we extend this to other frameworks we will need subclass this
class Config():
    def __init__(self):
        # default values
        self.batch_size = 32
        self.im_size = (256, 256) # w, h
        self.max_epoch = 40 # maximum number of epochs you can run
        self.num_class = 1
        self.trainset = {
            # ratio being 0 means not using any healthy samples
            "pn_ratio": 0, 
        }

        self.preprocess = {
            "cropping": False,
            "resize": True,
            "flip": False, # horizontal flip
        }

        self.network = {
            "name": "SEResUNet", 
            "reconstruct": False,
            "weight_decay": False,
        }
        self.optimizer = None
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
        
        if "trainset" in cf_dict.keys():
            self.trainset = cf_dict["trainset"]

        # this should be a list in the json file 
        # indicating the order of preprocessing?
        assert "normalize" in cf_dict["preprocess"].keys() # This is explicitly required now
        for key in cf_dict["preprocess"]:
            self.preprocess[key] = cf_dict["preprocess"][key]

        if "network" in cf_dict:
            for key in cf_dict["network"]:
                self.network[key] = cf_dict["network"][key]
        self.optimizer = cf_dict["optimizer"]
        if "multiclass_loss" in cf_dict.keys():
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
        # if "eval" in cf_dict.keys():
        #     self.eval = cf_dict["eval"]

class EvalConfig():
    def __init__(self):
        self.person_status = None
    
    def load_from_json(self, cf_file):
        with open(cf_file) as f:
            cf_dict = json.load(f)
        self.person_status = cf_dict["person_status"]