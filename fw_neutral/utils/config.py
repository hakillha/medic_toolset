import json

class Config():
    def __init__(self):
        # default values
        self.batch_size = 32
        self.im_size = (256, 256)
        self.max_epoch = 40 # maximum number of epochs you can run
        self.num_class = 1
        self.optimizer = None
        self.multiclass_loss = None

    def load_from_json(self, cf_file):
        with open(cf_file) as f:
            cf_dict = json.load(f)
        # or use the give a default value dictionary indexing interface
        self.batch_size = cf_dict["batch_size"]
        self.im_size = tuple([dim for dim in cf_dict["im_size"]])
        self.max_epoch = cf_dict["max_epoch"]
        self.num_class = cf_dict["num_class"]
        self.optimizer = cf_dict["optimizer"]
        self.multiclass_loss = cf_dict["multiclass_loss"]
        assert self.multiclass_loss in ["softmax", "sigmoid"]