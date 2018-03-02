'''Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new process methods, etc...
by stratospark
from:
https://raw.githubusercontent.com/stratospark/food-101-keras/master/tools/image_gen_extended.py
see:
http://blog.stratospark.com/deep-learning-applied-food-classification-deep-learning-keras.html
https://github.com/fchollet/keras/issues/3338
'''
from __future__ import absolute_import
from __future__ import print_function

from collections import Counter
import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import sys
from functools import partial, reduce
import threading
import copy
import inspect
import types
import multiprocessing as mp

from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
from PIL import Image

from crop_breast import Cropper
from sample_regularly_inbreast import  classify_patch

def _get_histogram_thr_(img, qmin=0.3, qmax=0.7, bins=25):

    lq = (qmin)/2.0
    uq = 1 - (1-qmax)/2.0
    #print(lq, uq)
    lthr, uthr = np.percentile(img.ravel(), np.r_[lq, uq]*100.0)
    mask0 = (img.ravel() < uthr) & (img.ravel()>lthr)

    yh, xh = np.histogram(img.ravel()[mask0], bins=bins)
    cumh = np.cumsum(yh)/sum(yh)
    cumh -= cumh[0]
    cumh /= cumh[-1]
    #
    cumh += lq
    cumh *= uq
    mask = (cumh > qmin) & (cumh < qmax)
    try:
        thr_ind = np.argmax(mask) + np.argmin(yh[mask])
    except ValueError as ee:
        print("shape", img.shape)
        print("mask", sum(mask))
        print("cumh", cumh)
        print("max", max(img.ravel()))
        raise ee

    thr = xh[thr_ind]
    return thr

def threshold_wi_range(img, qmin=0.2, qmax=0.7, bins=25):
    """threshold an image, assuming that qmin to qmax fraction of pixels are below the threshold
    qmin"""
    try:
        thr = _get_histogram_thr_(img, qmin=qmin, qmax=qmax, bins=bins)
    except ValueError as ee:
        thr = _get_histogram_thr_(img, qmin=qmin, qmax=qmax, bins=bins*64)
    return np.maximum(0, img-thr)


def dicom_image_reader(filepath,
                       target_mode=None,
                       target_size=None,
                       dim_ordering='tf', 
                       **kwargs):
    import gzip
    import dicom
    if filepath.endswith("gz"):
        opener = gzip.open
    else:
        opener = open
    with opener(filepath, 'r') as fh:
        ds = dicom.read_file(fh)
    return img_to_array(ds.pixel_array, dim_ordering=dim_ordering)


def random_rotation(x, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0., rng=None):
    theta = np.pi / 180 * rng.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0., rng=None):
    h, w = x.shape[row_index], x.shape[col_index]
    tx = rng.uniform(-hrg, hrg) * h
    ty = rng.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shear(x, intensity, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0., rng=None):
    shear = rng.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0., rng=None):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = rng.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass


def random_channel_shift(x, intensity, channel_index=0, rng=None):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + rng.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0,
                    fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel,
                            final_affine_matrix,
                            final_offset, order=0, 
                            mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def expand_dims(imga, classes=1):
    if classes==1:
        if imga.ndim==2:
            imga = np.expand_dims(imga, axis=0)
    else:
        pass
    return imga

def img_read_and_preprocess(path, streamer=None,
                         target_size=[350, 200],
                         transform=None,
                         dtype="float32",
                         pad_mode='median',
                         zoomin = 1.5,
                         zoomout=2.0,
                         trim_margin=5,
                         downsamplingthr = 2.0,
                         lowerthreshold = False,
                         qnorm_unity = None,
                         seed=None, rng=np.random, classes=None):
    """Apply a pipeline of image reading and transformation:
    1. read an image,
    2. crops and/or downsamples it to a desired size:
        `target_size * zoomin`
    3. applies transformation `process(image, rng)` with a specified seed
    4. post-apply central cropping to size `target_size` if zoomin > 1.0
    5. expand dimensions if image is 2D to (?,?,1)
    """
    if seed:
        rng.seed(seed)

    if not classes:
        if target_size is not None:
            if len(target_size)<=2 or target_size[2]==1:
                classes = 1
            elif len(target_size)>2 and target_size[2]>1:
                classes = target_size[2]
        else:
            classes=1

    if classes==1:
        dtype=dtype or "float32"
    else:
        dtype=dtype or "uint8"
    if zoomin is None:
        zoomin=1.0

    if target_size is not None:
        read_target_size = list(target_size)
        if zoomin>1.0:
            for nn in [0,1]:
                read_target_size[nn] = int(round(zoomin * target_size[nn]))
    else:
         read_target_size = target_size

    imga = pil_image_reader_croppad(path, streamer=streamer,
                                    dtype=dtype,
                                    downsamplingthr=downsamplingthr,
                                    target_size=read_target_size,
                                    zoomout=zoomout,
                                    trim_margin = trim_margin,
                                    pad_mode=pad_mode)
    #print("1 min={}\tmax={}".format( min(imga.ravel()), max(imga.ravel())  ))
    if transform is not None:
        imga = transform(imga, rng=np.random)
    #print("2 min={}\tmax={}".format( min(imga.ravel()), max(imga.ravel())  ))
    imga = np.asarray(imga, dtype=dtype)
    if lowerthreshold:
        imga = threshold_wi_range(imga)
    if qnorm_unity is not None:
        imga = qnorm_to_unity(imga, qnorm_unity)
    #print("3 min={}\tmax={}".format( min(imga.ravel()), max(imga.ravel())  ))
    return imga

def _post_process_(x, classes,
                   transform=lambda x:x, 
                   qnorm_unity=None,
                   target_size=None,
                   pad_mode='median',
                   make_onehot = True,
                   flatten=False):

    if transform is not None:
        x = transform(x)
    if target_size is not None:
        #print(x.shape)
        x = crop_pad_center(x, target_size, pad_mode=pad_mode)
    #print("4.a min={}\tmax={}\tch={}".format( min(x.ravel()), max(x.ravel()), x.shape[-1]))
    #print("classes", classes)
    if classes>1:
        #print( "n(1)", sum(x.ravel()>0) )
        #x = simplify_channel(x)
        #print("counter:",Counter(list(x.ravel())))
        if make_onehot:
            x = onehot(x, n_classes=classes, dtype='uint8')
            #print("sums:", np.apply_over_axes(np.sum, x, (0,1)))
    elif (qnorm_unity is not None) and qnorm_unity>0:
        # only for original images, not for masks
        x = qnorm_to_unity(x, qnorm_unity)
    #print("4.b min={}\tmax={}\tch={}".format( min(x.ravel()), max(x.ravel()), x.shape[-1]))
    x = partial(expand_dims, classes=classes)(x)
    #print("4.c min={}\tmax={}\tch={}".format( min(x.ravel()), max(x.ravel()), x.shape[-1]))
    if flatten:
        x = x.reshape(-1, x.shape[-1])
    return x


def array_to_img(x, dim_ordering='tf', mode=None, scale=True):
    x = x.copy()
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3 and mode == 'RGB':
        return Image.fromarray(x.astype('uint8'), mode)
    elif x.shape[2] == 1 and mode == 'L':
        return Image.fromarray(x[:, :, 0].astype('uint8'), mode)
    elif mode:
        return Image.fromarray(x, mode)
    else:
        raise Exception('Unsupported array shape: ', x.shape)


def img_to_array(x, dim_ordering='tf', dtype='float32'):
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    if dtype:
        x = np.asarray(x, dtype=dtype)
    else:
        x = np.asarray(x)
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x

def load_img(path, target_mode=None, target_size=None):
    img = Image.open(path)
    # don't convert if  not asked too!
    if target_mode:
        img = img.convert(target_mode)
    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.ANTIALIAS)
    return img

def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]

def pil_image_reader(filepath, target_mode=None, target_size=None, dim_ordering='tf', **kwargs):
    img = load_img(filepath, target_mode=target_mode, target_size=target_size)
    return img_to_array(img, dim_ordering=dim_ordering)

def qnorm_to_unity(x, q=0.05):
    if q is None:
        return x
    if q<0.5:
        q = 1.0 - q
    max_ = np.percentile(x, 100*q)
    if max_ !=0:
        x /= max_
    return x


def standardize(x,
                dim_ordering='th',
                rescale=False,
                featurewise_center=False,
                samplewise_center=False,
                qnorm_unity = None,
                featurewise_std_normalization=False,
                mean=None, std=None,
                samplewise_std_normalization=False,
                zca_whitening=False, principal_components=None,
                featurewise_standardize_axis=None,
                samplewise_standardize_axis=None,
                fitting=False,
                verbose=0,
                config={},
                **kwargs):
    '''

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        featurewise_standardize_axis: axis along which to perform feature-wise center and std normalization.
        samplewise_standardize_axis: axis along which to to perform sample-wise center and std normalization.
        zca_whitening: apply ZCA whitening.
        qnorm_unity: normalize upper `qnorm_unity` quantiles to 1.0

    '''
    if fitting:
        if '_X' in config:
            # add data to _X array
            config['_X'][config['_iX']] = x
            config['_iX'] +=1
            # if verbose and config.has_key('_fit_progressbar'):
                # config['_fit_progressbar'].update(config['_iX'], force=(config['_iX']==fitting))

            # the array (_X) is ready to fit
            if config['_iX'] >= fitting:
                X = config['_X'].astype('float32')
                del config['_X']
                del config['_iX']
                if featurewise_center or featurewise_std_normalization:
                    featurewise_standardize_axis = featurewise_standardize_axis or 0
                    if type(featurewise_standardize_axis) is int:
                        featurewise_standardize_axis = (featurewise_standardize_axis, )
                    assert 0 in featurewise_standardize_axis, 'feature-wise standardize axis should include 0'

                if featurewise_center:
                    mean = np.mean(X, axis=featurewise_standardize_axis, keepdims=True)
                    config['mean'] = np.squeeze(mean, axis=0)
                    X -= mean

                if featurewise_std_normalization:
                    std = np.std(X, axis=featurewise_standardize_axis, keepdims=True)
                    config['std'] = np.squeeze(std, axis=0)
                    X /= (std + 1e-7)

                if zca_whitening:
                    flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
                    sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
                    U, S, V = linalg.svd(sigma)
                    config['principal_components'] = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
                if verbose:
                    del config['_fit_progressbar']
        else:
            # start a new fitting, fitting = total sample number
            config['_X'] = np.zeros((fitting,)+x.shape)
            config['_iX'] = 0
            config['_X'][config['_iX']] = x
            config['_iX'] +=1
            # if verbose:
                # config['_fit_progressbar'] = Progbar(target=fitting, verbose=verbose)
        return x

    if rescale:
        x *= rescale

    # x is a single image, so it doesn't have image number at index 0
    if dim_ordering == 'th':
        channel_index = 0
    if dim_ordering == 'tf':
        channel_index = 2

    samplewise_standardize_axis = samplewise_standardize_axis or channel_index
    if type(samplewise_standardize_axis) is int:
        samplewise_standardize_axis = (samplewise_standardize_axis, )

    if samplewise_center:
        x -= np.mean(x, axis=samplewise_standardize_axis, keepdims=True)
    if samplewise_std_normalization:
        x /= (np.std(x, axis=samplewise_standardize_axis, keepdims=True) + 1e-7)
    if qnorm_unity:
        x = qnorm_to_unity(x, qnorm_unity)

    if verbose:
        if (featurewise_center and mean is None) or (featurewise_std_normalization and std is None) or (zca_whitening and principal_components is None):
            print('WARNING: feature-wise standardization and zca whitening will be disabled, please run "fit" first.')

    if featurewise_center:
        if mean is not None:
            x -= mean
    if featurewise_std_normalization:
        if std is not None:
            x /= (std + 1e-7)

    if zca_whitening:
        if principal_components is not None:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x

def custom_crop(x, center_crop_size, **kwargs):
    pass

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh, :]

def random_crop(x, random_crop_size, sync_seed=None, rng=None, **kwargs):
    # np.random.seed(sync_seed)
    w, h = x.shape[0], x.shape[1]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    #print('w: {}, h: {}, rangew: {}, rangeh: {}'.format(w, h, rangew, rangeh))
    offsetw = 0 if rangew == 0 else rng.randint(rangew)
    offseth = 0 if rangeh == 0 else rng.randint(rangeh)
    return x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1], :]

from keras.applications.inception_v3 import preprocess_input as pp

def preprocess_input(x, rng=None, **kwargs):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    return pp(x)

def random_transform(x,
                     dim_ordering='tf',
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     rescale=None,
                     sync_seed=None,
                     rng=None,
                     **kwargs):
    '''

    # Arguments
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
    '''
    # rng.seed(sync_seed)

    x = x.astype('float32')
    # x is a single image, so it doesn't have image number at index 0
    if dim_ordering == 'th':
        img_channel_index = 0
        img_row_index = 1
        img_col_index = 2
    if dim_ordering == 'tf':
        img_channel_index = 2
        img_row_index = 0
        img_col_index = 1
    # use composition of homographies to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * rng.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = rng.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = rng.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_index]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if shear_range:
        shear = rng.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    if np.isscalar(zoom_range):
        zoom_range = [1 - zoom_range, 1 + zoom_range]
    elif len(zoom_range) == 2:
        zoom_range = [zoom_range[0], zoom_range[1]]
    else:
        raise Exception('zoom_range should be a float or '
                        'a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = rng.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    h, w = x.shape[img_row_index], x.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, img_channel_index,
                        fill_mode=fill_mode, cval=cval)
    if channel_shift_range != 0:
        x = random_channel_shift(x, channel_shift_range, img_channel_index, rng=rng)

    if horizontal_flip:
        if rng.rand() < 0.5:
            x = flip_axis(x, img_col_index)

    if vertical_flip:
        if rng.rand() < 0.5:
            x = flip_axis(x, img_row_index)

    # TODO:
    # barrel/fisheye

    #rng.seed()
    return x

###################################

class ImageDataGeneratorExtended(ImageDataGenerator):

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 qnorm_unity = None,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        super(ImageDataGeneratorExtended, self).__init__(
                    samplewise_center=samplewise_center,
                    featurewise_std_normalization=featurewise_std_normalization,
                    samplewise_std_normalization=samplewise_std_normalization,
                    zca_whitening=zca_whitening,
                    zca_epsilon=zca_epsilon,
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    shear_range=shear_range,
                    zoom_range=zoom_range,
                    channel_shift_range=channel_shift_range,
                    fill_mode=fill_mode,
                    cval=cval,
                    horizontal_flip=horizontal_flip,
                    vertical_flip=vertical_flip,
                    rescale=rescale,
                    preprocessing_function=preprocessing_function,
                    data_format=data_format,
                )
        self.config = copy.deepcopy(locals())
        self.config['config'] = self.config
        self.config['mean'] = None
        self.config['std'] = None
        self.config['principal_components'] = None
        self.config['rescale'] = rescale

        # if dim_ordering not in {'tf', 'th'}:
        #      raise Exception('dim_ordering should be "tf" (channel after row and '
        #                       'column) or "th" (channel before row and column). '
        #                       'Received arg: ', dim_ordering)
 
        self.__sync_seed = self.config.get('seed') or np.random.randint(0, 4294967295)


        
        self.default_pipeline = []
        #self.default_pipeline.append(self.random_transform)
        #self.default_pipeline.append(self.standardize)
        #self.default_pipeline.append(preprocess_input)
        self.default_pipeline.append(partial(random_transform, **self.config))
        self.default_pipeline.append(partial(standardize, **self.config))
        self.set_pipeline(self.default_pipeline)
        
    def process(self, x, rng):
        # get next sync_seed
        # np.random.seed(self.__sync_seed)
        #np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        #self.__sync_seed = np.random.randint(0, 4294967295)
        # __sync_seed = rng.randint(0, 4294967295)
        # # print(self.__sync_seed)
        # self.config['fitting'] = self.__fitting
        # try:
        #     del self.config['sync_seed']
        # except:
        #     pass
        #self.config['sync_seed'] = self.__sync_seed
        for p in self.__pipeline:
            x = p(x, rng=rng, **self.config)
        return x

    def __call__(self, x, rng=np.random):
        return self.process(x, rng=rng)

    @property
    def pipeline(self):
        return self.__pipeline

    def sync(self, image_data_generator):
        self.__sync_seed = image_data_generator.sync_seed
        return (self, image_data_generator)

    def set_pipeline(self, p):
        if p is None:
            self.__pipeline = self.default_pipeline
        elif type(p) is list:
            self.__pipeline = p
        else:
            raise Exception('invalid pipeline.')


###################################
class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N=None, batch_size=32, shuffle=False, seed=None):
        if N==None:
            N = self.N
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if self.batch_index == 0:
                self.index_array = np.arange(N)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    self.index_array = np.random.permutation(N)
                    if seed is not None:
                        np.random.seed()

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (self.index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __add__(self, it):
        assert self.N == it.N
        assert self.batch_size == it.batch_size
        assert self.shuffle == it.shuffle
        seed = self.seed or np.random.randint(0, 4294967295)
        it.total_batches_seen = self.total_batches_seen
        self.index_generator = self._flow_index(self.N, self.batch_size, self.shuffle, seed)
        it.index_generator = it._flow_index(it.N, it.batch_size, it.shuffle, seed)
        if (sys.version_info > (3, 0)):
            iter_zip = zip
        else:
            from itertools import izip
            iter_zip = izip
        return iter_zip(self, it)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

#############################################
class BatchAccumulator(Iterator):
    """
    Input:
        filenames      -- arbitrary list of inputs
        read_process   -- arbitrary function (or list of functions) 
                          that take the items from the list of inputs and map them to
                          outputs of type numpy.array
        batch_size     --
        shuffle
        seed
    """
    def __init__(self, filenames, read_process, batch_size=4, shuffle=True, seed=None):
        self.filenames = filenames
        self.batch_index = 0
        self.read_process = read_process
        super(BatchAccumulator, self).__init__(len(self.filenames), batch_size, shuffle, seed)
        #
        self.index_generator = self._flow_index(self.N, self.batch_size, self.shuffle, seed)
        self._reader_generator = []
        self.maxint = np.iinfo(np.int32).max
        if seed:
            np.random.seed(seed)
        
    def reset(self):
        self.batch_index = 0
    
    def get_seed(self):
        return np.random.randint(self.maxint)

    def next(self):
        """Produce a batch from a simple generator of arbitrary samples
        A sample might have zero, one, or more labels.
        Final shape of images and labels are inferred 
        from the size of the first sample in each batch.
        """
        _new_generator_flag = False
        sample_count = 0
        with self.lock:
            index_array, current_index, batch_size = next(self.index_generator)
        # batch_size = self.batch_size
        batch_size = len(index_array)
        # print(len(index_array))
        for sample_count, ii in enumerate(index_array):
            ff = self.filenames[ii]
            "set the seed here!"
            seed = self.get_seed()
            gen_outs = self.read_process(ff, seed = seed)
            if sample_count == 0:
                # initialize based on the shapes within the first sample
                batch_out = ([None]*len(gen_outs))
                for nn in range(len(gen_outs)):
                    batch_out[nn] = np.zeros((batch_size,) + gen_outs[nn].shape)
            for nn, xx in enumerate(gen_outs):
                batch_out[nn][sample_count] = xx
            if sample_count >= self.batch_size:
                break
        return batch_out
###################################
def get_shape(img):
    if type(img) in (np.array, np.ndarray):
        return img.shape
    elif "PIL." in str(type(img)):
        h, w = img.size[1], img.size[0]
        if len(img.size)==2:
            return h, w
        else:
            return h, w, img.size[2:] 

def trim_fixed(img, margin=5):
    "input: PIL.Image"
    width, height = img.size
    halfmargin = margin//2
    _halfmargin = margin - margin//2
    return img.crop([_halfmargin, _halfmargin, width-halfmargin, height-halfmargin])

def pil_image_reader_croppad(filepath, target_mode=None,
                             target_size=None, 
                             downsamplingthr = 2.0,
                             zoomout = None,
                             pad_mode = 'reflect',
                             trim_margin = 5,
                             dim_ordering='tf',
                             streamer=None,
                             dtype='uint8', **kwargs):
    """Load an image from file and apply cropping and/or padding to a desired size
    + if zoomout is specified:
        first downsample by factor of `zoomout` or `1/zoomout` and then apply crop/pad
    + otherwise if downsamplingthr is specified and the image is significantly larger 
    than the requested size
    (i.e. original area exceeds the requested area by factor of `downsamplingthr`)
    the image is downsampled 
    """
    if streamer is None:
        img = Image.open(filepath)
    else:
        # if path is an s3 object or so
        img = Image.open(streamer(filepath))
    # don't convert if  not asked too!
    if target_mode:
        img = img.convert(target_mode)
    #img = load_img(filepath, target_mode=target_mode, target_size=None)
    img = trim_fixed(img, margin=trim_margin)
    h,w = get_shape(img)
    if zoomout is not None and (zoomout!=1.0):
        if zoomout>1.0:
            zoomout = 1.0/zoomout
        resample = Image.NEAREST if dtype=='uint8' else Image.BILINEAR
        img = np.asarray(img.resize((round(w*zoomout), round(h*zoomout)),
                    resample=resample))
        #print("range after resize", min(img.ravel()), max(img.ravel()) )
        if target_size:
            img = crop_pad_center(img, target_size, pad_mode=pad_mode)
    elif target_size:
        if downsamplingthr and downsamplingthr*np.prod(target_size) < (h*w):
            aspectratio = w / h
            img = np.asarray(img.resize((round(target_size[0]*aspectratio), target_size[0],),
            resample=Image.BILINEAR))
        else:
            img = np.asarray(img)
        img = crop_pad_center(img, target_size, pad_mode=pad_mode)
    else:
        img = np.asarray(img)
    
    #print("0 min={}\tmax={}".format( min(img.ravel()), max(img.ravel())  ))
    img = img_to_array(img, dim_ordering=dim_ordering, dtype=dtype)
    return img


def crop_pad_center(img, size, pad_mode='reflect', centre=None):
    """crop and/or pad image to the specified size
    """
    assert len(img.shape)>1, "empty image"
    pad_seq, crop_seq = _get_crop_pad_seq_(img, size=size, centre=centre)
    # print("pad_seq", pad_seq)
    # print("img.shape", img.shape)
    try:
        img = img[crop_seq]
        out = np.pad(img, pad_seq, mode=pad_mode)
    except (IndexError,TypeError) as ee:
        print("img.shape", img.shape, sep='\t', file=sys.stderr)
        print("target size", size, sep='\t', file=sys.stderr)
        print("pad_seq", pad_seq, sep='\t', file=sys.stderr)
        print("crop_seq", crop_seq, sep='\t', file=sys.stderr)
        raise ee
    return out

def _get_crop_pad_seq_(img, size, centre = None):
    """a helper function for crop_pad_center
    IN :        |--------x--------|
            
    OUT:    |------------.----X-----------------|
            0           cin
    cout:       |<------>|
    
    """
    sz_in = img.shape
    if centre is None:
        centre = [x//2 for x in sz_in]
    pad_seq = []
    crop_seq = []
    for sin, sout, cin in zip(sz_in, size, centre):
        sz_diff = (sout - sin)
        cout = sout//2
        # positive: pad
        # negative: crop
        if sz_diff<0:
            pad_seq.append((0,0))
            left_diff = cout - cin
            right_diff = cout + cin - sin
            #print("left_diff", left_diff, "right_diff", right_diff)
            if left_diff>0:
                right_diff += left_diff # -(-left_crop)
                left_diff = 0
            elif right_diff>0:
                left_diff += right_diff # -(-right_crop)
                right_diff = 0
            crop_seq.append(slice(-left_diff, sout-left_diff, 1))
        else:
            crop_seq.append(slice(None))
            cin = sin//2
            left_diff = cout - cin
            right_diff = cout + cin - sin
            pad_seq.append((left_diff, right_diff))
    # adjust for difference in dimensionality
    ndim_diff = len(img.shape) - len(size)
    if ndim_diff>0:
        for nn in range(ndim_diff):
            crop_seq.append(slice(None))
            pad_seq.append((0,0))
    return pad_seq, crop_seq

def joint_crop_mask_centre(aa, bb,
                           target_size=[64,64],
                           shift_range = 0.0,
                           non_mask_rate = 0.0,
                           pad_mode='median',
                           classes = (1,2,3,4,6),
                           alpha=5e-3,
                           seed=None, **kwargs):

    if non_mask_rate>0.0 and np.random.rand()<non_mask_rate:
        #in_size = np.asarray(bb.shape[:2], dtype=int)
        imc = Cropper(aa)
        #imc.crop()
        imc.crop_approx(axis=0, alpha=alpha)
        imc.crop_approx(axis=1, alpha=alpha)
        in_size = np.asarray(imc.im.shape, dtype=int)[:2]
        border = np.asarray(target_size, dtype=int)//2
        #print("in_size",in_size)
        #print("border", border)
        in_size -= border 
        y, x = np.asarray(np.round(border + in_size*np.random.rand(2)), dtype=int)
        y += imc.bbox[0][0]
        x += imc.bbox[1][0]
        print("non-mask\tx={}\ty={}".format(x,y))
    else: 
        y, x, height, width = get_mask_box_centre(bb, classes)
        print("feature x={}\ty={}\th={}\tw={}".format(x,y, height, width))
        if shift_range > 0:
            y_r, x_r = shift_range*(1 - 2*np.random.rand(2))
            y += int(np.round(y_r*height/2))
            x += int(np.round(x_r*width/2))
        x = min(max(0,x), bb.shape[1])
        y = min(max(0,y), bb.shape[0])
    #print("x={}\ty={}".format(x,y))
    #print("aa: min={}\tmax={}".format( min(aa.ravel()), max(aa.ravel())  ))
    #print("bb: min={}\tmax={}".format( min(bb.ravel()), max(bb.ravel())  ))
    aa = crop_pad_center(aa, target_size, centre=[y,x], pad_mode=pad_mode)
    bb = crop_pad_center(bb, target_size, centre=[y,x], pad_mode=pad_mode)
    print("classes within the box", np.bincount(bb.ravel()))
    #print("bb=1: {}\tbb=2:{}".format( sum(bb.ravel()==1), max(bb.ravel()==2)  ))
    return aa, bb


def obj_bboxes(msk,):
    im = msk.astype(int)
    label_im, num = ndi.label(im)
    slices = ndi.find_objects(label_im)
    
    valid_slices = np.where(np.bincount(label_im[msk]))
    if len(valid_slices):
        valid_slices = valid_slices[0]
        #print("valid_slices", valid_slices)
        #centroids = ndi.measurements.center_of_mass(im, label_im,
        #                                                valid_slices)
        slices = [slices[x-1] for x in valid_slices]
        return [get_slice_row_bbox(x) for x in  slices]

def get_slice_row_bbox(slicerow):
    xstart, xstop = slicerow[0].start, slicerow[0].stop
    ystart, ystop = slicerow[1].start, slicerow[1].stop
    return xstart, ystart, xstop, ystop
    #return ystart, xstart, ystop, xstop

def get_mask_box_centre(img, classes=None):
    print('looking for classes:', classes)
    if classes is None:
        mask = img>0
    else:
        mask = reduce(np.maximum, [img==cc for cc in classes])

    #print("mask pixels", sum(mask.ravel()), img.shape, img.dtype)
    #mask = Image.fromarray(np.asarray(np.squeeze(mask), dtype="uint8"))
    bboxes = obj_bboxes(mask)
    #box = mask.getbbox()
    if len(bboxes) ==0:
        warntxt = ("no bounding box found within the image; " 
                        + str(np.bincount(img.ravel())))
        print(warntxt)
        return (mask.shape[0]//2, mask.shape[1]//2, 120, 120)
    box = bboxes[np.random.randint(len(bboxes))]
    if box is not None:
        #left, upper, right, lower = box
        upper, left, lower, right= box
        x_c = (left+right)//2
        y_c = (upper+lower)//2
        x_len = right - left
        y_len = lower - upper
    else:
        x_len, y_len = mask.size
        x_c = x_len // 2
        y_c = y_len // 2
    return (y_c, x_c, y_len, x_len)
#########################################
def img_read_and_process(path, process, target_size=[350, 200], 
                         dtype="float32",
                         zoomin = 1.5,
                         zoomout = 1.5,
                         downsamplingthr=None,
                         pad_mode = 'reflect',
                         trim=3,
                         seed=None, rng=np.random, classes=None):
    """Apply a pipeline of image reading and transformation:
    1. reads an image and subsamples by factor of `zoomout`
    2. crops and/or downsamples it to a desired size:
        `target_size * zoomin`
    3. applies transformation `process(image, rng)` with a specified seed
    4. post-apply central cropping to size `target_size` if zoomin > 1.0
    5. expand dimensions if image is 2D to (?,?,1)
    """
    if seed:
        rng.seed(seed)
    
    if not classes:
        if len(target_size)<=2 or target_size[2]==1:
            classes = 1
        elif len(target_size)>2 and target_size[2]>1:
            classes = target_size[2]
    
    if classes==1:
        dtype=dtype or "float32"
    else:
        dtype=dtype or "uint8"
    
    read_target_size = list(target_size)
    if (zoomin is not None) and (zoomin>1.0):
        for nn in [0,1]:
            read_target_size[nn] = int(round(zoomin * target_size[nn]))
    imga = pil_image_reader_croppad(path, dtype=dtype,
                                    downsamplingthr=downsamplingthr,
                                    zoomout=zoomout,
                                    pad_mode=pad_mode,
                                    target_size=read_target_size)
    if (zoomin is not None) and (zoomin>1.0):
        imga = center_crop(imga, target_size)

    if classes==1:
        if imga.ndim==2:
            imga = np.expand_dims(imga, axis=0)
    else:
        pass
    
    imga = process(imga, rng=np.random)
    imga = np.asarray(imga, dtype=dtype)
    return imga

def _check_lambdas_(process, num, name="process"):
    ve = "keyword argument `{}` must be either a callable or a list of callables".format(name)
    ve = ValueError(ve)
    if hasattr(process, '__call__'):
        process = [process]*num
    elif not hasattr(process, '__len__'):
        raise ve
    elif len(process) == num:
        if not all((hasattr(proc, '__call__') for proc in process)):
            raise ve
        pass
    elif len(process) < num and len(process)==2:
        process = ([process[0]]*(num-1)) + [process[1]]
    else:
        raise ve
    return process

def process_sample(*paths,
                    process=lambda x: x,
                    joint_process=None,
                    post_process=None,
                    seed=None,
                    rng=np.random
                    ):
    """For each _item_ in `paths` apply a transformation `process`,
    making sure that the `process` receives the same random seed.

    Inputs:
    - paths   -- any type that `process` expects to receive.
    - process -- a function (callable) or a list/tuple of functions
    - seed    -- a seed for random number generator (RNG) or None
                 if None, an internal random seed is generated
                 and used to re-seed the RNG for each item pair of
                 `paths` and `process`
    - rng     -- random number generator that can be seeded by calling
                 rng.seed(seed)
    """
    seed = seed or rng.randint(2**32)
    process = _check_lambdas_(process, len(paths))
    ###################################
    outputs = []
    for proc, path in zip(process, paths):
        #print(os.path.basename(path), seed)
        rng.seed(seed)
        outputs.append(proc(path))


    if joint_process is not None:
        try:
            outputs = joint_process(*outputs, seed=seed)
        except Exception as err:
            print("error while processing:", *paths, file=sys.stderr, sep="\n")
            raise err
        outputs = list(outputs)

    print("4 min={}\tmax={}".format( min(outputs[1].ravel()), max(outputs[1].ravel())  ))
    if post_process is not None:
        post_process = _check_lambdas_(post_process, len(paths), 
                                       name="post_process")
        for nn, (proc, out) in enumerate(zip(post_process, outputs)):
            #print("post_process input", out.shape)
            rng.seed(seed)
            try:
                #print( "n(1)", sum(out.ravel()>0) )
                outputs[nn] = proc(out)
                #if outputs[nn].shape[-1]==3:
                #    print( "n(1)", sum(outputs[nn][:,:,1].ravel()>0) )
            except Exception as ee:
                print("error on", paths[nn])
                raise ee

    print("5 min={}\tmax={}".format( min(outputs[1].ravel()), max(outputs[1].ravel())  ))
    return outputs

####################################
def onehot(xx, n_classes=8, dtype=bool):
    "apply one-hot encoding to an integer-valued image"
    yy = np.zeros((xx.shape[:-1]+(n_classes,)), dtype=bool)
    for cc in range(n_classes):
        yy[:,:,cc][xx[:,:,0]==cc] = True
    if dtype is not bool:
        yy = np.asarray(yy, dtype=dtype)
    return yy

def simplify_channel(x):
    if x>2:
        return 0
    else:
        return x

simplify_channel = np.vectorize(simplify_channel)
    
def int_to_mass_calc(xx):
    n_classes=3
    yy = np.zeros((xx.shape[:-1]+(n_classes,)), dtype=bool)
    yy[:,:,1][(xx[:,:,0]>0)&(xx[:,:,0]<=5)] = True
    yy[:,:,2][(xx[:,:,0]>4)] = True
    return yy
####################################
def wr_process_sample_onehot(paths, seed=None,
                             augment=lambda x:x,
                             streamer=None,
                             target_size=(64,64),
                             n_classes=1,
                             trim_margin=10,
                             shift_range = 0,
                             qnorm_unity=1e-3,
                             pad_mode='median',
                             zoomout=1.0,
                             zoomin=1.0,
                             flatten_output=False,
                             lowerthreshold=False,
                             non_mask_rate=0.0,
                             rng_joint=None,
                             make_onehot=True,
                             ):
    #print("target_size", target_size)
    prelim_target_size = [int(round(x*zoomin)) for x in target_size]
#     print("prelim_target_size", prelim_target_size)
    process = [
        lambda x: img_read_and_preprocess(x,
                                        target_size=None,
                                        streamer=streamer,
                                        pad_mode = 'median',
                                        trim_margin=trim_margin,
                                        dtype='float32',
                                        zoomout=zoomout,
                                        qnorm_unity=qnorm_unity,
                                        lowerthreshold=lowerthreshold,
                                        zoomin=zoomin),
        lambda x: img_read_and_preprocess(x,
                                       target_size=None,
                                       streamer=streamer,
                                       zoomout=zoomout,
                                       zoomin=zoomin,
                                       pad_mode = 'constant',
                                       trim_margin=trim_margin,
                                       dtype='uint8')
              ]
    return process_sample(*paths,
                          process=process,
                          seed=seed,
                          joint_process=partial(joint_crop_mask_centre,
                                                target_size=prelim_target_size,
                                                shift_range=shift_range,
                                                non_mask_rate = non_mask_rate),
                          post_process=[partial(_post_process_,
                                                transform=augment,
                                                target_size=target_size,
                                                pad_mode=pad_mode,
                                                classes=1,
                                                ),
                                         partial(_post_process_,
                                                 make_onehot=make_onehot,
                                                 transform=augment,
                                                 target_size=target_size,
                                                 pad_mode=pad_mode,
                                                 classes=n_classes,
                                                 flatten=flatten_output)]
                        )
                        

def get_class_from_pair_blank(aa, bb, thr=0.05):
    if thr is None:
        thr = 0.0
    non_blank_area = (aa>0).ravel().mean()
    if non_blank_area<thr:
        return aa, bb.shape[-1] # np.r_[0,0,0,1] # blank
    else:
        chsums = np.zeros(bb.shape[-1])
        for cc in range(1,bb.shape[-1]):
            chsums[cc] = bb[:,:, cc].ravel().sum()
        label = np.argmax(chsums)
        if sum(chsums)==0:
            label=0
        #print("chsums", chsums)
        #print("argmax", label)
        return aa, label # np.r_[0,1,0,0] # mass
    return aa, 0#  np.r_[1,0,0,0]


def wr_process_sample_class(paths, **kwargs):
    classif_args = ["areathr", "mass_patch_frac", "n_total_pix_mask", "mincalc", "mass_tot_frac"]
    classif_dict = {}
    for kk in classif_args:
        vv = kwargs.get(kk)
        if vv is not None:
            classif_dict[kk] = vv
            kwargs.pop(kk)

    aa, mask = wr_process_sample_onehot(paths, **kwargs)
    return aa, classify_patch(mask, **classif_dict)
                      # n_total_pix_mask=0,
                      # mincalc = 3,
                      # mass_patch_frac=areathr,
                      # mass_tot_frac = 1/4)

    #return get_class_from_pair_blank(aa,bb, thr=thr)

def wr_process_sample_class_segm(paths, **kwargs):
    classif_args = ["mass_patch_frac", "n_total_pix_mask", "mincalc", "mass_tot_frac"]
    classif_dict = {}
    for kk in classif_args:
        vv = kwargs.get(kk)
        if vv is not None:
            classif_dict[kk] = vv
            kwargs.pop(kk)

    aa, mask  = wr_process_sample_onehot(paths, make_onehot=False, **kwargs)
    #print('='*30)
    #aa, yy =  get_class_from_pair_blank(aa,bb, thr=thr)
    yy = classify_patch(mask, **classif_dict)
    mask = onehot(mask, n_classes=kwargs["n_classes"], dtype='uint8')
    print("mask sums:", np.apply_over_axes(np.sum, mask, (0,1)))

    #aa = histeq(aa, mask = aa>0)
    return aa, mask, yy, paths[0]

