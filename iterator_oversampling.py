from keras.utils.data_utils import Sequence
import numpy as np
import sys
from copy import deepcopy
import threading

class Iterator(Sequence):
    """Abstract base class for image data iterators.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed,
                 postprocessing_function=None,
                 stratify=None,
                 stratified_shuffle=False,
                 oversampling=True,
                 sampling_factor=None):
        self.stratify = stratify
        self.oversampling = oversampling
        self.stratified_shuffle = stratified_shuffle
        self.sampling_factor = sampling_factor
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.postprocessing_function = postprocessing_function
        self.index_generator = self._flow_index()
        self.final_num = self.n
        if self.stratify is not None:
            self._prep_stratified()
            if self.oversampling:
                self.final_num = len(self.orig_index_array)
                #sum(self.class_size>0) * max(self.class_size)
                print("Final number of samples:\t%d" % self.final_num)
        else:
            self.orig_index_array = np.arange(self.n)

    def _prep_stratified(self):
        if not hasattr(self, '_classes'):
            self._classes = np.unique(self.stratify)
        self.class_size = np.bincount(self.stratify)
        
        self.class_inds = {}
        for cc in self._classes:
            mask = self.stratify == cc
            self.class_inds[cc] = np.where(mask)[0]


        ex_per_class = {}
        if self.sampling_factor is not None:
            for cc, ff in zip(self._classes, self.sampling_factor):
                ex_per_class[cc] = int(np.ceil(ff*len(self.class_inds[cc])))
            print("ex_per_class", ex_per_class)
        elif self.oversampling:
            max_class_size = max(self.class_size)
            for cc in list(set(self._classes)):
                ex_per_class[cc] = max_class_size
        else:
            for cc in self._classes:
                ex_per_class[cc] = len(self.class_inds[cc])
                
        #print("ex_per_class", ex_per_class)
        #print("self._classes", self._classes, file=sys.stderr)
        self.orig_index_array = []
        for cc,ss in enumerate(self.class_size):
                if ss==0:
                    continue
                if not cc in ex_per_class:
                    print("class not in `ex_per_class`", cc, file=sys.stderr)
                    continue
                self.orig_index_array.append(
                    np.random.choice(self.class_inds[cc],
                                    size=ex_per_class[cc],
                                    replace=ss<ex_per_class[cc])
                                )
#         np.random.permutation(self.n)
        if self.oversampling and self.sampling_factor is None:
            self.orig_index_array = np.stack(self.orig_index_array).T.ravel()
        else:
            tmp = deepcopy(self.orig_index_array)
            self.orig_index_array = []
            for row in tmp:
                self.orig_index_array.extend(row)
#             if self.shuffle and self.stratified_shuffle:
#                 tmp = [list(np.random.permutation(x)) for x in tmp]
#             else:
#                 tmp = [list(x) for x in tmp]
#             self.orig_index_array = []
            
#             nempty = 0
#             while nempty<len(tmp):
#                 nempty = 0
#                 for row in enumerate(tmp):
#                     try:
#                         elem = row.pop()
#                         self.orig_index_array.append(elem)
#                     except IndexError as ee:
#                         nempty +=1
            self.orig_index_array = np.asarray(np.random.permutation(self.orig_index_array))

            
    def _set_index_array(self):
        self.index_array = self.orig_index_array.copy()
        if self.shuffle and not self.stratified_shuffle:
            self.index_array = np.random.permutation(self.orig_index_array.copy())
                
            
    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0
        
    @property
    def iter_per_epoch(self):
        return int(np.ceil(self.final_num / self.batch_size))

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()

        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.final_num
            if self.final_num > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
#             print("current_index", current_index)
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]


    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
