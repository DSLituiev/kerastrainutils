#import pandas as pd
#from inception import get_model, get_num_files, get_class_weights, freezer
#from checkpoint_utils import CSVWallClockLogger, AttrDict, copy_weights_layer_by_layer, get_chckpt_info
#from PIL import Image
from functools import partial
from itertools import product

import keras
#from keras.utils import np_utils
from keras.layers import Lambda
from keras import backend as K
#from keras import optimizers
#from image import ImageDataGenerator
#from inception import get_model, get_num_files, get_class_weights, freezer
#from checkpoint_utils import CSVWallClockLogger, AttrDict, copy_weights_layer_by_layer, get_chckpt_info
#from PIL import Image
#import keras
#from keras.utils import np_utils
#from keras.layers import Lambda
#from keras import backend as K
#from keras import optimizers
#from image import ImageDataGenerator
#from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
#from memmap_tables import read_array
#from collections import Counter

#from model_surgery import capture_summary
import tensorflow as tf

def weighted_binary_crossentropy(y_true, y_pred, weight=0.5):
    bxe = K.binary_crossentropy(y_pred, y_true)
    weight_vector = y_true * weight + (1. - y_true) * (1 - weight)
    weight_vector = weight_vector / K.sum(weight_vector)
    wbxe = weight_vector * bxe
    return K.mean( wbxe, axis=1)

def w_categorical_crossentropy(weights):
    def _w_categorical_crossentropy(y_true, y_pred, weights):
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.expand_dims(y_pred_max, 1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):

            final_mask += (K.cast(weights[c_t, c_p],K.floatx()) *
                           K.cast(y_pred_max_mat[:, c_p] ,K.floatx()) *
                           K.cast(y_true[:, c_t],K.floatx())
                          )
        return K.categorical_crossentropy(y_pred, y_true) * final_mask

    ncce = partial(_w_categorical_crossentropy, weights=weights)
    ncce.__name__ ='w_categorical_crossentropy'
    return ncce

def weighted_sparse_softmax_cross_entropy_with_logits(y_true, logits, weights=[]):
    input_shape = y_true.shape
    output_shape=[x.value for x in input_shape[:-1]]
    y_true = Lambda(lambda x: x[:,:,:,0], output_shape=output_shape)(y_true)
    y_true = tf.cast(y_true, tf.int32)
    raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=logits)
    print("loss shape", raw_loss.shape)
    print("number of class weights:", len(weights))

    w_sum = sum(weights)
    weights = [float(ww)/w_sum for ww  in weights]

    tot_loss = []
    npixels = []
    for ii,ww in enumerate(weights):
        mask_ = tf.equal(y_true,ii)
        npix = tf.reduce_sum(tf.cast(mask_, tf.int64))
        condition = tf.greater(npix, 0)
        npixels.append(npix)
        
        masked_loss = tf.boolean_mask(raw_loss, mask_)
        masked_loss = tf.cond(condition,
                           lambda : masked_loss,
                           lambda :  tf.Variable([0.0], dtype=tf.float32))
        loss_ = ww * tf.reduce_mean(masked_loss)
        tot_loss.append(loss_)

    rescale = (tf.cast(tf.reduce_sum(npixels), tf.float32) / 
               tf.reduce_sum([tf.cast(nn, tf.float32)*ww for nn,ww in zip(npixels, weights)])
               )
    tot_loss = [ls*rescale for ls in (tot_loss)]
    tot_loss = tf.reduce_sum(tot_loss)
    return tot_loss


def sparse_softmax_cross_entropy_with_logits(y_true, logits, alpha=0.1):
    input_shape = y_true.shape
    output_shape=[x.value for x in input_shape[:-1]]
    y_true = Lambda(lambda x: x[:,:,:,0], output_shape=output_shape)(y_true)
    y_true = tf.cast(y_true, tf.int32)
    out = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=logits)
    print("loss shape", out.shape)
    mask_bg = tf.equal(y_true,5)
    mask_fg = tf.cast( tf.logical_not(mask_bg), tf.float32)
    mask_bg = tf.cast( mask_bg, tf.float32)

    fg_c = tf.reduce_sum(mask_fg)
    bg_c = tf.reduce_sum(mask_bg)

    tot_c = fg_c+bg_c

    fg = mask_fg * (tot_c/fg_c)
    bg = mask_bg * (tot_c/bg_c)
    #ca = tf.logical_or(tf.equal(y_true,1),tf.equal(y_true,2) )
    #be = tf.logical_or(tf.equal(y_true,3),tf.equal(y_true,4) )
    #fgloss = tf.boolean_mask(out, fg)
    #print("fgloss shape", fgloss.shape)
    #fgloss = tf.boolean_mask(out, fg)
    loss = alpha*out*bg + (1-alpha)*out*fg
    #bgloss = tf.boolean_mask(out, bg)
    #loss = alpha*tf.reduce_sum(bgloss) + (1-alpha)*tf.reduce_sum(fgloss)
    #print("loss shape", out.shape)
    return loss


def sparse_accuracy(y_true, logits):
    y_true_ = tf.cast(y_true, tf.int64)[:,:,:,0]
    pred = tf.argmax(logits, axis=-1)
    match = tf.equal(y_true_, pred)
    match = tf.cast(match, tf.float32)
    return match


def sparse_accuracy_fg(y_true, logits, bgind = 5):
    match = sparse_accuracy(y_true, logits)
    
    y_true_ = tf.cast(y_true, tf.int64)[:,:,:,0]
    mask_bg = tf.equal(y_true_, bgind)
    mask_fg = tf.logical_not(mask_bg)
    
    masked_match = tf.boolean_mask(match, mask_fg)

    condition = tf.greater(tf.reduce_sum(tf.cast(mask_fg, tf.int64)), 0)
    masked_match = tf.cond(condition,
                           lambda : masked_match,
                           lambda : tf.Variable([0.0], dtype=tf.float32))
    
    
    mask_fg = tf.cast(mask_fg, tf.float32)
    return tf.reduce_mean(masked_match)


def sparse_accuracy_masscalc(y_true, logits, bgind = 5, normind=0):
    match = sparse_accuracy(y_true, logits)
    
    y_true_ = tf.cast(y_true, tf.int64)[:,:,:,0]
    mask_bg = tf.equal(y_true_, bgind) 
    mask_norm = tf.equal(y_true_, normind)
    mask_bg = tf.logical_or(mask_bg, mask_norm)
    mask_fg = tf.logical_not(mask_bg)
    
    masked_match = tf.boolean_mask(match, mask_fg)

    condition = tf.greater(tf.reduce_sum(tf.cast(mask_fg, tf.int64)), 0)
    masked_match = tf.cond(condition,
                           lambda : masked_match,
                           lambda : tf.Variable([0.0], dtype=tf.float32))
    
    
    mask_fg = tf.cast(mask_fg, tf.float32)
    return tf.reduce_mean(masked_match)


########################################33
############################################################
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    """PFA, prob false alert for binary classifier"""
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N


def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    """P_TA prob true alerts for binary classifier"""
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def auroc(y_true, y_pred):
    """AUC for a binary classifier
    by @isaacgerg from https://github.com/fchollet/keras/issues/3230
    """
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)


############################################################
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

"""
from tensorflow.python.ops import control_flow_ops
@static_vars(stream_vars=None)
def auc(y_true, y_pred, curve='ROC'):
    value, update_op = tf.contrib.metrics.streaming_auc(
        y_pred, y_true, curve=curve, name='auc_'+curve.lower())

    auc_roc.stream_vars = [i for i in tf.local_variables() if i.name.split('/')[0] == 'auc_roc']
    return control_flow_ops.with_dependencies([update_op], value)
"""

############################################################
#   Losses per class
############################################################

def acc_cl(y_true, y_pred, cl=0):
    if len(y_true.shape) == 1 or any(1==x for x in y_true.shape):
        print("sparse")
        y_true = K.max(y_true, axis=-1)
    else:  # one-hot case
        print("one-hot")
        y_true = K.cast(K.argmax(y_true, axis=-1), K.floatx())
    y_pred = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    mask = K.equal(y_true, cl)
    mask = K.cast(mask, K.floatx())
    #y_true = tf.boolean_mask(y_true, mask)
    #y_pred = tf.boolean_mask(y_pred, mask)
    y_true = y_true * mask
    y_pred = y_pred * mask
    per_entry = K.cast(K.equal(y_true, y_pred),
                  K.floatx())
    per_entry = per_entry * mask
    return K.switch(K.equal(0.0, K.sum(mask)),
                    0.0, K.sum(per_entry) / K.sum(mask))

def acc_0(y_true, y_pred):
    return acc_cl(y_true, y_pred, cl=0)

def acc_1(y_true, y_pred):
    return acc_cl(y_true, y_pred, cl=1)

def acc_2(y_true, y_pred):
    return acc_cl(y_true, y_pred, cl=2)

def acc_3(y_true, y_pred):   return acc_cl(y_true, y_pred, cl=3)

def acc_4(y_true, y_pred):   return acc_cl(y_true, y_pred, cl=4)

def acc_5(y_true, y_pred):   return acc_cl(y_true, y_pred, cl=5)

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.losses.losses import compute_weighted_loss
import keras
from keras import backend as K

def pairwise_binary_logsigmoid(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    #reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
    ):

    with ops.name_scope(scope, "absolute_difference",
                      (predictions, labels, weights)) as scope:
        mask_pos = tf.equal(labels, 1)
        mask_neg = tf.equal(labels, 0)

        yhat_pos = tf.boolean_mask(predictions, mask_pos)
        yhat_neg = tf.boolean_mask(predictions, mask_neg)

        yhat_diff = (tf.reshape(yhat_pos, (-1,1)) -
                     tf.reshape(yhat_neg, (1,-1))
                    )

        losses = tf.log_sigmoid(-yhat_diff)
        print("losses",losses.dtype)
        loss = tf.reduce_sum(losses)
        #util.add_loss(loss, loss_collection)
        return loss
        """
        return compute_weighted_loss(
            losses, weights, scope, loss_collection,
            #reduction=reduction
            )
            """


def keras_tf_pairwise_binary_logsigmoid(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    #reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
    ):

    with ops.name_scope(scope, "absolute_difference",
                      (predictions, labels, weights)) as scope:
        mask_pos = K.equal(labels, 1)
        mask_neg = K.equal(labels, 0)

        yhat_pos = K.tf.boolean_mask(predictions, mask_pos)
        yhat_neg = K.tf.boolean_mask(predictions, mask_neg)

        score_diff = (K.reshape(yhat_pos, (-1,1)) -
                     K.reshape(yhat_neg, (1,-1))
                    )

        #losses = K.tf.log_sigmoid(-yhat_diff)
        losses = -K.softplus(score_diff)
        print("losses",losses.dtype)
        return  K.sum(losses*weights) #K.tf.reduce_sum(losses)



