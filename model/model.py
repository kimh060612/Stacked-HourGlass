import tensorflow as tf
from tensorflow import keras as tfk
from layers import HourGlass, ResidualLayer

class StackedHourGlassNetwork(tfk.Model):
    def __init__(self, InputShape = (256, 256, 3), numStack = 4, numResidual = 1, numHeatMap = 16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numStack = numStack
        self.numResidual = numResidual
        self.numHeatMap = numHeatMap
        self.Input = tfk.layers.Input(shape=InputShape)
        self.conv1 = tfk.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="SAME", kernel_initializer='he_normal')
        self.Batchnorm1 = tfk.layers.BatchNormalization(momentum=0.99)
        self.ReLU1 = tfk.layers.ReLU()
        self.ResNet1 = ResidualLayer(InputChannel=64, OutputChannel=128)
        self.MaxPool1 = tfk.layers.MaxPool2D(pool_size=2, strides=2)
        self.ResNet2 = ResidualLayer(InputChannel=128, OutputChannel=128)
        self.ResNet3 = ResidualLayer(InputChannel=128, OutputChannel=256)
        self.StackedHourGlass = []
        for i in range(self.numStack):
            
            pass

    def call(self, Input):
        pass

        


