import tensorflow as tf
from tensorflow import keras as tfk
from model.layers import HourGlass

class StackedHourGlassNetwork(tfk.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    







