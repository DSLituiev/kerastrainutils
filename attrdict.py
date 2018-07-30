import os
import sys
import csv
import six
import numpy as np
import re
import time

from collections import OrderedDict
from collections import Counter
from collections import Iterable

####################################
from hashlib import md5
import json
import yaml
try:
    import pyaml
except:
    print("no `pyaml` module found. Using `yaml` instead", file=sys.stderr)



class AttrDict(dict):
    """A dictionary allowing to retrieve values by class attribute syntax
    additionally supports `to_yaml()` serialization and `md5` hashing
    """
    def _convert_fun_(x):
        if hasattr(x, "__call__"):
            return x.__name__
        else:
            return x

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def load(cls, filename):
        obj = cls.__new__(cls)  # Does not call __init__
        with open(filename) as fh:
            kwargs= yaml.load(fh)
            super(AttrDict, obj).__init__(**kwargs)
            obj.__dict__ = obj 
            return obj

    def to_yaml(self, filename, use_pyaml=True):
        #safedict = {kk: _convert_fun_(vv) for kk,vv in self.__dict__}
        with open(filename, "w+") as outfh:
            if use_pyaml:
                try:
                    pyaml.dump(self.__dict__, outfh)
                except yaml.representer.RepresenterError as ee:
                    yaml.dump(self.__dict__, outfh, default_flow_style=False)
            else:
                yaml.dump(self.__dict__, outfh, default_flow_style=False)

    def to_json(self, filename):
        #safedict = {kk: _convert_fun_(vv) for kk,vv in self.__dict__}
        with open(filename, "w+") as outfh:
                json.dump(self.__dict__, outfh)

    @property
    def md5(self):
        txt = str([(kk, self.__dict__[kk]) for kk in sorted(self.__dict__.keys())])
        return md5(txt.encode()).hexdigest()


def keras_to_json(model, path, hash=False, name=None):
    if name is not None:
        model.name = name
    jstr = model.to_json()
    if hash:
        name = md5(jstr.encode()).hexdigest()
        name += ".json"
        path = os.path.join(path, name)

    with open(path, 'w') as f:
        f.write(jstr)

####################################
def get_chckpt_info(indir):
    if not os.path.isdir(indir):
        indir = os.path.dirname(indir)
    infile = os.path.join(indir, "checkpoint.info")
    with open(infile) as fh:
        info = AttrDict(yaml.load(fh))
    return info


####################################

def get_class_weights(datagen_val_output):
    counter = Counter(datagen_val_output.classes)
    print(counter)
    for kk,vv in counter.items():
        counter[kk] = vv+1

    max_val = float(max(counter.values()))

    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     
    return class_weights


class CheckpointParser():
    def __init__(self, filename, prefix = 'model', suffix = "\.([0-9]*)-([0-9,.]*).hdf5"):
        self.pattern = re.compile(prefix + suffix)
        self.filename = filename
        self.result = self.pattern.findall(filename)
        if len(self.result):
            self.result = self.result[0]
            self.epoch = int(self.result[0])
            self.loss = float(self.result[1])
        else:
            self.result = ""
            self.epoch = -1
            self.loss = None
        self.result = [self.epoch, self.loss]
        return
    
    def __call__(self):
        return self.result

def find_min_loss_checkpoint(indir):
    loss = np.inf
    filename = ""
    epoch = 0
    for ff in os.scandir(indir):
        chkp = CheckpointParser(ff.name)
        if chkp.loss<loss:
            loss = chkp.loss
            filename = chkp.filename
            epoch = chkp.epoch

    return filename, epoch

####################################

def tent_function(lrmin, lrmax, step):
    def _tent_(epoch):
        ep = (epoch % step) / step
        if ep > 0.5:
            return lrmin*ep + lrmax*(1-ep)
        else:
            return lrmin*(1-ep) + lrmax*ep
    return (_tent_)



def tent(x, slope=1, width=0.50):
    half = width/2
    xf, xi = np.modf(x/ (half))
    if xi % 2 ==0:
        return xf*slope
    else:
        return (1- xf)*slope

def lr_cyclic_schedule(epoch,
    lr_init = 1.0e-3,
    drop = 2/5,
    epochs_drop = 20,
    cycle_len = 200.0
    ):
    width = cycle_len/epochs_drop
    coef = tent(np.floor(epoch/epochs_drop), slope=width/2, width=width)
    lrate = coef
    lrate = lr_init * np.power(drop, coef)
    return lrate


