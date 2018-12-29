#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from sentencepiece import SentencePieceProcessor as spp
from sentencepiece import SentencePieceTrainer as spt
import os

def spm(path2data, size):
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

def encode(spm, sents, maxLen):
    sents = list(map(spm.EncodeAsIds, sents))
    arr = np.zeros((len(sents), maxLen))
    for idx, sent in enumerate(sents):
        arr[idx, :len(sent)] = sent
    return arr

def decode(spm, arr):
    sents = [spm.DecodeIds(list(map(int, arr[i,np.nonzero(arr[i,:])[0]]))) for i in range(len(arr))]
    return sents
