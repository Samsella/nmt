#!/usr/bin/python
# -*- coding: utf8 -*-

from utils2 import data
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    path2data = '../../../../data/europarl/de-en/'
    #path2data = '../data/de-en/'
    files = [path2data+'europarl-v7.de-en.de',
             path2data+'europarl-v7.de-en.en']
    d = data(files, size=16000)
    test = 0
    
    if not os.path.exists('data_16000.npz') or test:
        if test:
            if not os.path.exists('data_small.npz'):
                data_X, data_y, labels = d.get_data(test=test)
                np.savez_compressed('data_small', a=data_X, b=data_y, c=labels)
            else:
                dat = np.load('data_small.npz')
                data_X, data_y, labels = dat['a'], dat['b'], dat['c']
        else:
            data_X, data_y, labels = d.get_data(test=0)
            np.savez_compressed('data_16000', a=data_X, b=data_y, c=labels)
    else:
        dat = np.load('data_16000.npz')
        data_X, data_y, labels = dat['a'], dat['b'], dat['c']
    print(data_X.shape, data_y.shape, labels.shape)
    
    train_X, test_X, train_y, test_y, train_labels, test_labels= train_test_split(data_X, data_y, labels, test_size=0.002, random_state=15)
    
    def sl(y_true, y_pred):
        ''' wrap for sparse_categorical_crossentropy to bypass the adding
            of an extra dimension
        '''
        return scc(y_true, y_pred)

    callback_checkpoint = ModelCheckpoint(filepath='chpt{epoch:02d}_4001_1200.keras',
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
    #sn = SliceNet(vocab_size=4000, depth=1200)
    #sn.compile(adam, sl)
    #model = sn.model
 
    #if os.path.exists('chpts/chpt40_4001_1200.keras'):
    #    model.load_weights('chpts/chpt40_4001_1200.keras')
    #    print('Weights loaded')

    #model.fit(x=[train_X[:500000], train_y[:500000]], y=train_labels[:500000], batch_size=80, epochs=40,
     #         verbose=2, validation_split=0.002, callbacks=callbacks)


    #p = sn.predict(test_X)
    #decoded_de = d.decode(np.array(p), 'shared')
    original = d.decode(test_labels, 'shared')
    #with open('results/model_4001_1200.txt', 'w', encoding='utf8') as f:
    #    f.write('\n'.join(decoded_de))
    with open('results/original_de_16000.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(original))
train()    
