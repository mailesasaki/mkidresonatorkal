# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:29:21 2021

@author: autum
"""

import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model('/mnt/c/Users/autum/OneDrive/Desktop/Tensorflow Project/Tensorflow Models/saved_model4')

new_model.summary()

data = np.load('/mnt/c/Users/autum/OneDrive/Desktop/Tensorflow Project/Tensorflow Models/fullval_final_4.npz')

train_images = data['trainImages']
train_labels = data['trainLabels']
test_images = data['testImages']
test_labels = data['testLabels']

loss, acc = new_model.evaluate(test_images, test_labels)
