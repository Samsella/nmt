#!/usr/bin/python
# -*- coding: utf8 -*-

from utils import data
from model_2 import SliceNet
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os


def train():
    data_X, data_y = data().get_data()
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.1, random_state=13)

    if os.path.exists('SliceNet.h5') and 0:
        model = load_model('SliceNet.h5')
    else:
        sn = SliceNet()
        sn.compile('Adam', 'binary_crossentropy')
        model = sn.model
        model.save('SliceNet.h5')
    #model.compile('adam', 'binary_crossentropy')
    model.fit(train_X, train_y, 64, 1, 1, validation_data=(test_X, test_y))

train()
