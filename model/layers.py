import tensorflow as tf
from tensorflow import keras as tfk


class ResidualLayer(tfk.layers.Layer):
    def __init__(self):
        super(ResidualLayer, self).__init__(name="Residual")


    def call(self, input):
        
        pass