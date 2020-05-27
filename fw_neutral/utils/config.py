import json

class Config():
    def __init__(self):
        # default values
        self.batch_size = 32
        self.im_size = (256, 256) # w, h
        self.max_epoch = 40 # maximum number of epochs you can run
        self.num_class = 1
        self.optimizer = None
        self.loss = None
        self.eval = {"ct_interval": [-1400, 800], "norm_by_interval": False}
        self.preprocess = None

    def load_from_json(self, cf_file):
        with open(cf_file) as f:
            cf_dict = json.load(f)
        # or use the give a default value dictionary indexing interface
        self.batch_size = cf_dict["batch_size"]
        self.im_size = tuple([dim for dim in cf_dict["im_size"]])
        self.max_epoch = cf_dict["max_epoch"]
        self.num_class = cf_dict["num_class"]
        self.optimizer = cf_dict["optimizer"]
        if "multiclass_loss" in cf_dict.keys():
            self.loss = cf_dict["multiclass_loss"]
        else:
            self.loss = cf_dict["loss"]
        if self.num_class == 1:
            assert self.loss in ["sigmoid", "hbloss_dice_focal", "generalized_dice_loss", "dice_loss", "focal",
                "hbloss_dice_focal_v2", "hbloss_dice_ce"]
        else:
            assert self.loss in ["softmax", "sigmoid"]
        # use the default setting which is the old setting
        # for the old cfg files where eval configuration is missing
        # new cfg files have to overwrite this to work correctly
        if "eval" in cf_dict.keys():
            self.eval = cf_dict["eval"]
        # this should be a list in the json file 
        # indicating the order of preprocessing?
        if "preprocess" in cf_dict.keys():
            self.preprocess = cf_dict["preprocess"]