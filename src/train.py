#!/usr/bin/python
# -*- coding: utf8 -*-

from utils import data
import numpy as np
import tensorflow as tf
from model_2 import SliceNet
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.losses import sparse_categorical_crossentropy as scc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    data_X, data_y = data().get_data()
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.1, random_state=13)
    train_X2 = train_y

    def sl(y_true, y_pred):
        ''' wrap for sparse_categorical_crossentropy to bypass the adding
            of an extra dimension
        '''
        return scc(y_true, y_pred)


    if os.path.exists('SliceNet.h5') and 0:
        model = load_model('SliceNet.h5')
    else:
        sn = SliceNet()
        sn.compile('Adam', sl)
        model = sn.model
        #model.summary()
        model.save('SliceNet.h5')
    model.fit(x=[train_X, train_X2], y=train_y, batch_size=256, epochs=200, verbose=2)

train()
