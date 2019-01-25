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
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    path2data = '../../../../data/europarl/de-en/'
    #path2data = '../data/de-en/'
    files = [path2data+'europarl-v7.de-en.de',
             path2data+'europarl-v7.de-en.en']
    d = data(files, size=4000)
    start = d.spm_en.EncodeAsIds('<s>')
    end = d.spm_en.EncodeAsIds('</s>')

    data_X, data_y = d.get_data(test=1)
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.01, random_state=18)
    train_X2 = train_y
    print(train_X[150:151])
    print(d.decode(train_X[150:151], 'en'))
    print(train_y[150:151])
    print(d.decode(train_y[150:151], 'de'))

    def sl(y_true, y_pred):
        ''' wrap for sparse_categorical_crossentropy to bypass the adding
            of an extra dimension
        '''
        return scc(y_true, y_pred)

    callback_checkpoint = ModelCheckpoint(filepath='chpt.keras',
                                      monitor='val_loss',
                                      verbose=0,
                                      save_weights_only=True,
                                      save_best_only=True)


    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

    callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                   histogram_freq=0,
                                   write_graph=True)

    callbacks = [callback_checkpoint,
                 callback_tensorboard]

    if os.path.exists('SliceNet.h5') and 0:
        model = load_model('SliceNet.h5')
    else:
        sn = SliceNet(vocab_size=4000)
        sn.compile('Adam', sl)
        model = sn.model
        #model.summary()
        model.save('SliceNet.h5')
    model.fit(x=[data_X, data_y], y=data_y, batch_size=256, epochs=100,
              verbose=2, validation_split=0.01, callbacks=callbacks)


    #p = model.predict(test_X[2], start, end)
    #print(test_X[2])
    #print(d.decode(np.array(p), 'de'))

    #print(d.decode(train_X[0], 'en'))
    #print('\n\n')
    #print(d.decode(train_y[0], 'de'))

train()
