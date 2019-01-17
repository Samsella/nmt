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
    def __init__(self):
        self.spm = self.sp('../data/deu.txt', 4000)

    def sp(self, path2data, size):
        name = 'spm_{size}_{type}'.format(size=size, type='unigram')
        path2file = name + '.model'
        if not os.path.exists(path2file):
            spt.train('\
                --input={input} \
                --model_prefix={name} \
                --vocab_size={size}'.format(
                    input=path2data,
                    name=name,
                    size=size))
        spm = spp()
        spm.load(path2file)
        return spm

    def encode(self, spm, src, tgt, maxLen):
        # encode sentences
        s = list(map(spm.EncodeAsIds, src))
        t = list(map(spm.EncodeAsIds, tgt))
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

    def decode(self, spm, arr):
        sents = [spm.DecodeIds(list(map(int, arr[i,np.nonzero(arr[i,:])[0]]))) for i in range(len(arr))]
        return sents

    def create_data(self, spm):
        with open('../data/deu.txt', 'r', encoding='utf8') as f:
            de = []
            en = []
            for line in f:
                d,e = line.strip('\n').split('\t')
                de.append(d)
                en.append(e)
            arr_de, arr_en = self.encode(spm, de, en, 50)
        return arr_de, arr_en

    def get_data(self):
        a, b = self.create_data(self.spm)
        return a, b
