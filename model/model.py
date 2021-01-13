import tensorflow as tf
from tensorflow import keras as tfk
from layers import HourGlass, ResidualLayer

class StackedHourGlassNetwork(tfk.Model):
    def __init__(self, numStack = 4, numResidual = 1, numHeatMap = 16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numStack = numStack
        self.numResidual = numResidual
        self.numHeatMap = numHeatMap
        self.conv1 = tfk.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="SAME", kernel_initializer='he_normal')
        self.Batchnorm1 = tfk.layers.BatchNormalization(momentum=0.99)
        self.ReLU1 = tfk.layers.ReLU()
        self.ResNet1 = ResidualLayer(InputChannel=64, OutputChannel=128)
        self.MaxPool1 = tfk.layers.MaxPool2D(pool_size=2, strides=2)
        self.ResNet2 = ResidualLayer(InputChannel=128, OutputChannel=128)
        self.ResNet3 = ResidualLayer(InputChannel=128, OutputChannel=256)
        self.StackedHourGlass = []
        self.output = []
        for i in range(self.numStack):
            if i == 0:
                self.StackedHourGlass.append(HourGlass(3, 256, self.numResidual, 4))
            else :
                self.StackedHourGlass.append(HourGlass(256 , 256, self.numResidual, 4))
            for _ in range(self.numResidual):
                self.StackedHourGlass.append(ResidualLayer(256,256))
            
            self.StackedHourGlass.append(tfk.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer="he_normal"))
            self.StackedHourGlass.append(tfk.layers.BatchNormalization(momentum=0.9))
            self.StackedHourGlass.append(tfk.layers.ReLU())
            
            self.output.append(tfk.layers.Conv2D(filters=self.numHeatMap, kernel_size=(1, 1), strides=(1, 1), padding="SAME", kernel_initializer="he_normal"))

    def call(self, Input):
        
        Z = self.conv1(Input)
        Z = self.Batchnorm1(Z)
        Z = self.ReLU1(Z)
        Z = self.ResNet1(Z)
        Z = self.MaxPool1(Z)
        Z = self.ResNet2(Z)
        Z = self.ResNet3(Z)
        Y = []
        for i in range(self.numStack):
            Z = self.StackedHourGlass[i*self.numResidual](Z)
            for j in range(self.numResidual):
                Z = self.StackedHourGlass[i*self.numResidual + j + 1](Z)
            
            Z = self.StackedHourGlass[(i+1)*self.numResidual + 1](Z)
            Z = self.StackedHourGlass[(i+1)*self.numResidual + 2](Z)
            Z = self.StackedHourGlass[(i+1)*self.numResidual + 3](Z)
            
            Y.append(self.output[i](Z))

            

            pass

        


