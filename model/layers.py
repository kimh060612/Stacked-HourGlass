import tensorflow as tf
from tensorflow import keras as tfk

#BottleNeck을 발생 시키기 위한 ResNet 구조 
class ResidualLayer(tfk.layers.Layer):
    def __init__(self, InputChannel, OutputChannel, trainable=True, name="Residual", dtype=None, **kwargs):
        super().__init__(trainable=True, name="Residual", dtype=None, **kwargs)
        self.Reslayer = []
        self.InputChannel = InputChannel
        self.OutputChannel = OutputChannel
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

#HourGlass Layer 재현
class HourGlass(tfk.layers.Layer):
    # Input Channel, Output Channel, Residual Layer's Depth
    def __init__(self, Input, Output, numResidual, depth, trainable=True, name="HourGlass", dtype=None, **kwargs):
        super().__init__(trainable=True, name="HourGlass", dtype=None, **kwargs)
        self.depth = depth
        self.Input = Input
        self.Output = Output
        self.numResidual = numResidual
        self.HgNet = []
        self.MakeHoutGlass()

    #Declare Residual Network
    def MakeResidual(self):
        Layer = []
        Layer.append(ResidualLayer(InputChannel=self.Input, OutputChannel=self.Output))
        for i in range(1, self.numResidual):
            Layer.append(ResidualLayer(InputChannel=Layer[i-1].OutputChannel, OutputChannel=self.Output))
        return Layer

    #Declare HourGlass
    def MakeHoutGlass(self):
        for i in range(self.depth):
            Res = []
            for j in range(3):
                Res.append(self.MakeResidual())
            if i == 0:
                Res.append(ResidualLayer(self.Input, self.Output))
            self.HgNet.append(Res)

    #FeedForward HourGlass Network     
    def HourGlassForward(self, n, Input):
        up1 = Input
        for layers in self.HgNet[n-1][0]:
            up1 = layers(up1)
        Pool1 = tfk.layers.MaxPool2D(pool_size=2, strides=2)(Input)
        low1 = Pool1
        for layers in self.HgNet[n-1][1]:
            low1 = layers(low1)
        low2 = low1
        if n > 1:
            low2 = self.HourGlassForward(n-1, low2)
        else :
            low2 = self.HgNet[n-1][3](low2)
        low3 = low2
        for layers in self.HgNet[n-1][2]:
            low3 = layers(low3)
        up2 = tfk.layers.UpSampling2D(size=2)(low3)
        return up1 + up2
        
    def call(self, Input):
        return self.HourGlassForward(self.depth, Input)

    

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


"""class BatchNorm(tfk.layers.Layer):
    def __init__(self, eps=0.001,trainable=True, name="BatchNorm", dtype=None, **kwargs):
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
        return {**base_config, "eps" : self.eps}"""