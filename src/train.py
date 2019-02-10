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
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train():
    path2data = '../../../../data/europarl/de-en/'
    #path2data = '../data/de-en/'
    files = [path2data+'europarl-v7.de-en.de',
             path2data+'europarl-v7.de-en.en']
    d = data(files, size=4000)

    test = 1

    if not os.path.exists('data.npz') or test:
        if test:
            if not os.path.exists('data_small.npz'):
                data_X, data_y, labels = d.get_data(test=test)
                np.savez_compressed('data_small', a=data_X, b=data_y, c=labels)
            else:
                dat = np.load('data_small.npz')
                data_X, data_y, labels = dat['a'], dat['b'], dat['c']
        else:
            data_X, data_y, labels = d.get_data(test=0)
            np.savez_compressed('data', a=data_X, b=data_y, c=labels)
    else:
        dat = np.load('data.npz')
        data_X, data_y, labels = dat['a'], dat['b'], dat['c']

    train_X, test_X, train_y, test_y, train_labels, test_labels= train_test_split(data_X, data_y, labels, test_size=0.002, random_state=15)
    #print('\nTrain: \n')
    #print(train_X[150:151])
    #print(d.decode(train_X[150:151], 'en'))
    #print(train_y[150:151])
    #print(d.decode(train_y[150:151], 'de'))
    #print(train_labels[150:151])
    #print(d.decode(train_labels[150:151], 'de'))
    #print('\nTest: \n')
    #print(test_X.shape, test_X[149:150])
    #print(d.decode(test_X[149:150], 'en'))
    #print(test_y.shape, test_y[149:150])
    #print(d.decode(test_y[149:150], 'de'))
    #print(test_labels.shape, test_labels[149:150])
    #print(d.decode(test_labels[149:150], 'de'))


    def sl(y_true, y_pred):
        ''' wrap for sparse_categorical_crossentropy to bypass the adding
            of an extra dimension
        '''
        return scc(y_true, y_pred)

    callback_checkpoint = ModelCheckpoint(filepath='chpt{epoch:02d}_small.keras',
                                      monitor='val_loss',
                                      verbose=0,
                                      save_weights_only=True)


    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

    callback_tensorboard = TensorBoard(log_dir='./21_logs2/',
                                   histogram_freq=0,
                                   write_graph=True,
                                   update_freq=10000)

    callbacks = [callback_checkpoint,
                 callback_tensorboard]

    adam = Adam(lr=0.002)
    sn = SliceNet(vocab_size=4000, depth=300)
    sn.compile(adam, sl)
    model = sn.model

    if os.path.exists('chpt76_small.keras'):
        model.load_weights('chpt76_small.keras')
        print('Weights loaded')

   # model.fit(x=[train_X[:400000], train_y[:400000]], y=train_labels[:400000], batch_size=256, epochs=100,
    #          verbose=2, validation_split=0.002, callbacks=callbacks)

    print(test_X[:9,:])
    print(d.decode(np.array(test_X[:9,:]), 'en'))
    print(test_X[:9,:].shape, test_y[:9,:])
    print(d.decode(np.array(test_y[:9,:]), 'de'))
    p = sn.predict(test_X[:9,:])
    print(p)
    print(d.decode(np.array(p), 'de'))

train()
