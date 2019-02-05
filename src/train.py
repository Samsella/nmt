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
#from tensorflow.nn import sparse_softmax_cross_entropy_with_logits as tfscc
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def train():
    path2data = '../../../../data/europarl/de-en/'
    #path2data = '../data/de-en/'
    files = [path2data+'europarl-v7.de-en.de',
             path2data+'europarl-v7.de-en.en']
    d = data(files, size=4000)

    test = 1

    if not os.path.exists('data.npz') or test:
        data_X, data_y, labels = d.get_data(test=test)
        np.savez_compressed('data', a=data_X, b=data_y, c=labels)
    else:
        d = np.load('data.npz')
        data_X, data_y, labels = d['a'], d['b'], d['c']

    train_X, test_X, train_y, test_y, train_labels, test_labels= train_test_split(data_X, data_y, labels, test_size=0.002, random_state=13)
    print('\nTrain: \n')
    print(train_X[150:151])
    print(d.decode(train_X[150:151], 'en'))
    print(train_y[150:151])
    print(d.decode(train_y[150:151], 'de'))
    print(train_labels[150:151])
    print(d.decode(train_labels[150:151], 'de'))
    print('\nTest: \n')
    print(test_X[150:151])
    print(d.decode(test_X[150:151], 'en'))
    print(test_y[150:151])
    print(d.decode(test_y[150:151], 'de'))
    print(test_labels[150:151])
    print(d.decode(test_labels[150:151], 'de'))


    def sl(y_true, y_pred):
        ''' wrap for sparse_categorical_crossentropy to bypass the adding
            of an extra dimension
        '''
        return scc(y_true, y_pred)

    callback_checkpoint = ModelCheckpoint(filepath='chpt{epoch:02d}.keras',
                                      monitor='val_loss',
                                      verbose=0,
                                      save_weights_only=True)


    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

    callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                   histogram_freq=0,
                                   write_graph=True,
                                   update_freq=10000)

    callbacks = [callback_checkpoint,
                 callback_tensorboard]


    sn = SliceNet(vocab_size=4000)
    sn.compile('Adam', sl)
    model = sn.model

    if os.path.exists('chpt.keras'):
        model.load_weights('chpt.keras')
        print('Weights loaded')

    model.fit(x=[train_X, train_y], y=train_labels, batch_size=128, epochs=0,
              verbose=2, validation_split=0.002, callbacks=callbacks)

    print(test_y[1,:])
    print(d.decode(np.array(test_y[1,:]), 'de'))
    p = sn.predict(test_X[1,:])
    print(p)
    print(d.decode(np.array(p), 'de'))

    #print(d.decode(train_X[0], 'en'))
    #print('\n\n')
    #print(d.decode(train_y[0], 'de'))

train()
