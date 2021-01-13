import tensorflow as tf
from tensorflow import keras as tfk

class JointMSELoss(tfk.losses.Loss):
    def __init__(self, global_batch_size, reduction=losses_utils.ReductionV2.AUTO, name=None):
        super().__init__(reduction=reduction, name=name)
        self.global_batch_size = global_batch_size

    def call(self, y_true, y_pred):
        loss = 0
        for out in y_pred:
            weights = tf.cast(y_true > 0, dtype=tf.float32)*81 + 1
            loss += tf.math.reduce_mean(tf.math.square(y_true - y_pred)*weights)*(1./self.global_batch_size)
        return loss