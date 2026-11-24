import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas

from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import *

from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    EfficientNetB0,
    EfficientNetB3,
    InceptionV3,
)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Simple GPU test
a = tf.random.normal((2000, 2000))
b = tf.random.normal((2000, 2000))
c = tf.matmul(a, b)

print("Result device:", c.device)
