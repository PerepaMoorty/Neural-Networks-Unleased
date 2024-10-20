# Using Fashion MNSIT Dataset - Building a Neural Network with an accuracy of at least .8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

fashion_mnsit = tf.keras.datasets.fashion_mnsit
(trainImages, trainLabels), (testImages, testLabels) = fashion_mnsit.load_data()