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
from keras.callbacks import Callback
import keras
from keras import backend as K

####################################
from hashlib import md5
import json
import yaml
try:
    import pyaml
except:
    print("no `pyaml` module found. Using `yaml` instead", file=sys.stderr)


def freezer(model, base_trainable=True, freezefirstblocks=None):
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers

    if not base_trainable:
        freezefirstblocks = 0
        for nn, la in enumerate(model.layers):
            if type(la) is keras.layers.Concatenate:
                freezefirstblocks += 1

    if freezefirstblocks is not None:
        cc=0
        for nn, la in enumerate(model.layers):
            print("freezing: block {}\t layer {}".format(cc, la.name), sep='\t', file=sys.stderr)
            if type(la) is keras.layers.Concatenate:
                if cc==freezefirstblocks:
                    break
                cc+=1
            la.trainable = False
        for la in model.layers[nn:]:
            la.trainable = True
    return model


def check_num_frozen_blocks(model):
    blocks = [[]]
    for nn, la in enumerate(model.layers):
        blocks[-1].append(la.trainable)
        if type(la) is keras.layers.Concatenate:
            blocks.append([])
    return np.argmax([any(ff) for ff in blocks])


def unfreezer(epoch, model, step=100, startnblocks=11):
    if (epoch) % step==0 and epoch>0:
        freezefirstblocks = startnblocks - epoch//step + 1
        freezer(model, base_trainable=True, freezefirstblocks=freezefirstblocks)
        print("epoch:\t{}\tnum frozen blocks:\t{}".format(epoch, check_num_frozen_blocks(model)))
    return

def lr_from_dict(epoch, schedule):
    schedule = list(schedule)
    for kk, vv in schedule:
        if epoch>=kk[0] and ((kk[1] is None) or epoch<=kk[1]):
            return vv


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


def copy_layer_by_weight_names(lamo0, lamo1):
    mo0names = [w.name.split('/')[-1] for w  in lamo0.weights]
    mo1names = [w.name.split('/')[-1] for w  in lamo1.weights]
    mo0dict = dict(zip(mo0names, lamo0.get_weights()))
    lamo1.set_weights( [mo0dict[kk] for kk in mo1names])
    return

def copy_weights_layer_by_layer(to_model, from_model, strict=True):
    
    for nn, (lamo0, lamo1) in enumerate(zip(from_model.layers, to_model.layers)):
        ws0 = lamo0.weights
        ws1 = lamo1.weights
        if len(ws0)==len(ws1) and all([ww0.shape == ww1.shape for ww0,ww1 in zip(ws0, ws1)]):
            lamo1.set_weights(lamo0.get_weights())
            #print("weights set for {}".format(lamo1))
        else:
            print("shape mismatch for layer #{:d}:\t{}\t{}".format(nn, lamo0.name, lamo1.name))
            if strict:
                break
            elif lamo0.name == lamo1.name:
                copy_layer_by_weight_names(lamo0, lamo1)
            else:
                break
    return to_model

####################################

def tent(lrmin, lrmax, step):
    def _tent_(epoch):
        ep = (epoch % step) / step
        if ep > 0.5:
            return lrmin*ep + lrmax*(1-ep)
        else:
            return lrmin*(1-ep) + lrmax*ep
    return (_tent_)
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

class CSVWallClockLogger(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
        ```python
        csv_logger = CSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False, compute_auc=False):
        self.compute_auc = compute_auc
        self._prev_time_ = time.time()
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(CSVWallClockLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch', "time", "lr"] +
                                         self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        ##
        if self.compute_auc:
            if not batch_size:
                numfiles = len(datagen_val_output.filenames)
            steps = len(datagen_val_output.filenames)/prms['batch_size']
            out = model.predict_generator(self.datagen_val_output, 
                                          steps=steps)
        time_ = time.time()
        row_dict = OrderedDict(epoch=epoch,
                               time = time_ - self._prev_time_,
                               lr=self.lr,)
        self._prev_time_ = time_
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    @property
    def lr(self): 
        lr = float(K.get_value(self.model.optimizer.lr))
        return lr



    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

class UnfreezeScheduler(Callback):
    """layer unfreeze scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, schedule):
        super(UnfreezeScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        self.schedule(epoch, self.model)
        #K.set_value(self.model.optimizer.lr, lr)


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


