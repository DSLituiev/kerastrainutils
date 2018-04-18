import keras
#import tensorflow as tf
#from tensorflow.python.framework import ops
#from tensorflow.python.ops.losses.losses import compute_weighted_loss



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


def pairwise_binary_logsigmoid(
    labels, predictions, weights=1.0, 
    reduction=K.sum
    ):
    #ii,jj = np.tril_indices(int(y.shape[0]))
    ss = int(labels.shape[0])
    ii = np.repeat(np.arange(ss), ss)
    jj = np.tile(np.arange(ss), ss)
    #print(list(zip(ii,jj)))
    ii = tf.constant(ii, dtype = 'int64')
    jj = tf.constant(jj, dtype = 'int64')
    
    if hasattr(weights, '__len__'):
        weights = K.gather(weights, ii) * K.gather(weights, jj)
        
    score_diff = K.gather(predictions, ii) - K.gather(predictions, jj)
    mask =  K.greater(K.gather(labels, ii), K.gather(labels, jj))
    mask = K.cast(mask, K.floatx())
    print("mask", mask)
    #return score_diff*mask
    losses = -K.softplus(score_diff) * mask * weights
	if reduction in (K.sum, 'sum', sum):
	    return K.sum( losses )
	elif reduction is K.mean:
		return K.sum( losses ) / K.sum(mask)


if __name__ == '__main__':
    yhat = K.constant([0.1,0.7,1,0,1,0,1,0,1,0,0.0])
    y = K.constant([1,0,1,0,1,0,1,0,1,0,0])

    aucloss = pairwise_binary_logsigmoid(y, yhat)

    print("="*20)
    print("aucloss")
    print( K.eval(aucloss) )


