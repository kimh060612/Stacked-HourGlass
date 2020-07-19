import tensorflow as tf
from tensorflow import keras as tfk


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
    
    

class ResidualLayer(tfk.layers.Layer):
    def __init__(self, InputChannel, OutputChannel):
        super(ResidualLayer, self).__init__(name="Residual")
        self.Reslayer = []
        self.Reslayer.append(tfk.layers.BatchNormalization(momentum=0.99, epsilon=0.001))
        self.Reslayer.append(tfk.layers.Conv2D(int(OutputChannel/2), kernel_size=(1,1), strides=(1,1), padding="SAME", activation="relu", use_bias=True))
        self.Reslayer.append(tfk.layers.BatchNormalization(momentum=0.99, epsilon=0.001))
        self.Reslayer.append(tfk.layers.Conv2D(int(OutputChannel/2), kernel_size=(3,3), strides=(1,1), padding="SAME", activation="relu",use_bias=True))
        self.Reslayer.append(tfk.layers.BatchNormalization(momentum=0.99, epsilon=0.001))
        self.Reslayer.append(tfk.layers.Conv2D(OutputChannel, kernel_size=(1,1), strides=(1,1), padding="SAME", activation="relu",use_bias=True))
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
    def __init__(self, trainable=True, name=None, dtype=None):
        super(HourGlass, self).__init__(trainable=True, name=name, dtype=dtype)
        
