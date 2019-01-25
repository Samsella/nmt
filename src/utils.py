#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from sentencepiece import SentencePieceProcessor as spp
from sentencepiece import SentencePieceTrainer as spt
import os
import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt

class data():
    def __init__(self, path='../data/deu.txt', size=4000):
        self.paths = path
        self.path = self.create_path(path)
        self.spm_de = self.sp(size, 'de', path[0])
        self.spm_en = self.sp(size, 'en', path[1])
        self.spm_en.SetEncodeExtraOptions('bos:eos')

    def create_path(self, path):
        if type(path) == list:
            newpath = path[0]+'_extended.txt'
            with open(newpath, 'wb') as f:
                for path in path:
                    f.write(open(path, 'rb').read())
            path=newpath
        return path

    def sp(self, size, lang, path):
        name = 'spm_{size}_{type}_{data}_{lang}'.format(
            size=size, type='unigram', data='europarl', lang=lang)
        path2file = name + '.model'
        if not os.path.exists(path2file):
            spt.train('\
                --input={input} \
                --model_prefix={name} \
                --vocab_size={size} \
                --unk_id=3'.format(
                    input=self.path,
                    name=name,
                    size=size))
        spm = spp()
        spm.load(path2file)
        return spm

    def encode(self, src, tgt, maxLen):
        # encode sentences
        s = list(map(self.spm_en.EncodeAsIds, src))
        t = list(map(self.spm_de.EncodeAsIds, tgt))
        src, tgt = [], []
        # remove sentences longer than maxLen
        for s1, s2 in zip(s,t):
            if (max(len(s1), len(s2)) > maxLen):
                continue
            src.append(s1)
            tgt.append(s2)
        # create arrays
        arr_src = np.zeros((len(src), maxLen))
        arr_tgt = np.zeros((len(tgt), maxLen))
        # fill arrays
        for idx, (sent1, sent2) in enumerate(zip(src, tgt)):
            arr_src[idx, :len(sent1)] = sent1
            arr_tgt[idx, :len(sent2)] = sent2
        return arr_src, arr_tgt

    def decode(self, arr, lang):
        if lang == 'de':
            sents = [self.spm_de.DecodeIds(list(map(int, arr[i,np.nonzero(arr[i,:])[0]]))) for i in range(len(arr))]
        if lang == 'en':
            sents = [self.spm_en.DecodeIds(list(map(int, arr[i,np.nonzero(arr[i,:])[0]]))) for i in range(len(arr))]

        return sents

    def create_data(self, test):
        if type(self.paths) == list:
            en = [line for line in open(self.paths[1], encoding='utf8')]
            de = [line for line in open(self.paths[0], encoding='utf8')]
            if test:
                en = en[:5000]
                de = de[:5000]
        else:
            with open(self.path, 'r', encoding='utf8') as f:
                en = []
                de = []
                for line in f:
                    d,e = line.strip('\n').split('\t')
                    en.append(d)
                    de.append(e)
        arr_en, arr_de = self.encode(en, de, 40)
        return arr_en, arr_de

    def get_data(self, test=0):
        a, b = self.create_data(test)
        return a.astype(int), b.astype(int)


#path2data = '../data/de-en/'
#files = [path2data+'europarl-v7.de-en.de',
#         path2data+'europarl-v7.de-en.en']
#d = data(files, size=8000)
#a, b = d.get_data()
#c = d.decode(b[5000:5002], 'en')
#print(c)
#c = d.decode(b[5000:5002], 'de')
#print(c)
