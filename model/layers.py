import tensorflow as tf
from tensorflow import keras as tfk

class BatchNorm(tfk.layers.Layer):
    def __init__(self, eps=0.001,trainable=True, name=None, dtype=None, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.eps = eps
    
    def build(self, BatchInputShape):
        self.gamma = self.add_weight(name="gamma", shape=BatchInputShape[-1:], initializer="ones")
        self.beta = self.add_weight(name="beta", shape=BatchInputShape[-1:], initializer="zeros")
        super().build(BatchInputShape)

    def call(self, X):
        Mean, Var = tf.nn.moments(X, axes=-1, keep_dims=True)
        return self.gamma * (X - Mean)/(tf.sqrt(Var + self.eps)) + self.beta

    def compute_output_shape(self, BatchInputShape):
        return BatchInputShape

    def get_config(self):
        base_config = super.get_config()
        return {**base_config, "eps" : self.eps}

class ResidualLayer(tfk.layers.Layer):
    def __init__(self, InputChannel, OutputChannel):
        super(ResidualLayer, self).__init__(name="Residual")
        self.Reslayer = []
        self.Reslayer.append(tfk.layers.BatchNormalization(momentum=0.99, epsilon=0.001))
        self.Reslayer.append(tfk.layers.Conv2D(int(OutputChannel/2), kernel_size=(1,1), strides=(1,1), padding="VALID", use_bias=True))
        self.Reslayer.append(tfk.layers.BatchNormalization(momentum=0.99, epsilon=0.001))
        self.Reslayer.append(tfk.layers.ReLU())
        self.Reslayer.append(tfk.layers.Conv2D(int(OutputChannel/2), kernel_size=(3,3), strides=(1,1), padding="VALID",use_bias=True))
        self.Reslayer.append(tfk.layers.BatchNormalization(momentum=0.99, epsilon=0.001))
        self.Reslayer.append(tfk.layers.ReLU())
        self.Reslayer.append(tfk.layers.Conv2D(OutputChannel, kernel_size=(1,1), strides=(1,1), padding="VALID", activation="relu" ,use_bias=True))
        self.MidConvLayer = tfk.layers.Conv2D(OutputChannel, kernel_size=(1,1), strides=(1,1), use_bias=True)
        if InputChannel == OutputChannel:
            self.IsMidConvLayer = False
        else :
            self.IsMidConvLayer = True

    def call(self, Input):
        if self.IsMidConvLayer:
            ResidualLayer = self.MidConvLayer(Input)
        else :
            ResidualLayer = Input
        Z = Input
        for layers in self.Reslayer:
            Z = layers(Z)
        return Z + ResidualLayer

class HourGlass(tfk.layers.Layer):
    def __init__(self):
        super(HourGlass, self).__init__()
        self.HourGlass = []
        self.HourGlass.append()

    def call(self):

        pass

    




"""class ConvBlock(tfk.layers.Layer):
    def __init__(self, InputChannel, OutputChannel, kernel_size = (3,3), stride = (1,1), IsReLU = True, IsBias = True, IsNormalization = True):
        super(ConvBlock, self).__init__(trainable=True, name="ConvNet")
        self.Layers = []
        # ReLU?
        if IsReLU :
            self.Layers.append(tfk.layers.Conv2D(OutputChannel, kernel_size, strides=stride, padding="SAME", use_bias= IsBias, activation='relu'))
        else :
            self.Layers.append(tfk.layers.Conv2D(OutputChannel, kernel_size=kernel_size, strides=stride, padding="SAME", use_bias=IsBias))
        # Batchnorm?
        if IsNormalization :
            self.Layers.append(tfk.layers.BatchNormalization(momentum=0.99, epsilon=0.001), False)
        else :
            pass
     
    def call(self, Input):

        Z = Input
        for layer in self.Layers:
            Z = layer(Z)
        return Z"""