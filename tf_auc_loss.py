

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.losses.losses import compute_weighted_loss

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
        return tf.reduce_sum(losses)
        """
        return compute_weighted_loss(
            losses, weights, scope, loss_collection,
            #reduction=reduction
            )
            """
   #   util.add_loss(loss, loss_collection)

def keras_pairwise_binary_logsigmoid(
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

        yhat_diff = (K.reshape(yhat_pos, (-1,1)) -
                     K.reshape(yhat_neg, (1,-1))
                    )

        #losses = K.tf.log_sigmoid(-yhat_diff)
        losses = -K.softplus(yhat_diff)
        print("losses",losses.dtype)
        return K.tf.reduce_sum(losses)

if __name__ == '__main__':
    yhat = tf.constant([0.1,0.7,1,0,1,0,1,0,1,0,0.0])
    y = tf.constant([1,0,1,0,1,0,1,0,1,0,0])
    # Start tf session
    sess = tf.Session()
    aucloss = pairwise_binary_logsigmoid(y, yhat)

    print("="*20)
    print("aucloss")
    print(sess.run(aucloss))


