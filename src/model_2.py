#!usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
import keras as K
import numpy as np
from keras_layer_normalization import LayerNormalization

scope = tf.variable_scope
KL = K.layers
KB = K.backend

class conv_step():
    def __init__(self, filters, length, pad='same', dil=1 ,name=''):
        self.act  = KL.Activation('relu', name=name+'_Act')
        self.conv = KL.SeparableConv1D(filters, length, padding=pad, dilation_rate=dil, name=name+'_SepConv')
        self.norm = LayerNormalization(name=name+'_Norm')

    def __call__(self, x):
        return self.norm(self.conv(self.act(x)))


class conv_block():
    def __init__(self, name, train, depth=100):
        self.conv1   = conv_step(depth, 3, name=name+'_1')
        self.conv2   = conv_step(100, 3, name=name+'_2')
        self.add1    = KL.Add(name=name+'_Add_1')
        self.conv3   = conv_step(depth, 15,name=name+'_3')
        self.conv4   = conv_step(100, 15, dil=4,name=name+'_4')
        self.add2    = KL.Add(name=name+'_Add_2')
        self.dropout = KL.Dropout(0.5, name=name+'_Dropout')
        self.train   = train

    def __call__(self, x):
        y   = self.add1([x, self.conv2(self.conv1(x))])
        out = self.add2([x, self.conv4(self.conv3(x))])
        if self.train:
            out = self.dropout(out)
        return out

class attend():
    def __init__(self, name, depth=100):
        self.coeff = KL.Lambda(lambda x: x * 1/np.sqrt(depth), name=name+'_Coeff')
        self.conv1 = conv_step(depth, 5, 'same', 1, name=name+'_1')
        self.conv2 = conv_step(depth, 5, 'same', 2, name=name+'_2')
        self.T     = KL.Lambda(lambda x: KB.permute_dimensions(x, (0,2,1)), name=name+'_Transpose')
        self.dot1  = KL.Dot(axes=(2,1), name=name+'_Dot_1')
        self.softm = KL.Softmax(name=name+'_Softmax')
        self.dot2  = KL.Dot(axes=1, name=name+'_Dot_2')

    def __call__(self, x, y):
        c = self.coeff(x)
        x_T = self.T(x)
        y = self.conv2(self.conv1(y))
        out = self.dot2([self.softm(self.dot1([y, x_T])), c])
        return out

class encoder():
    def __init__(self, train):
        self.c1 = conv_block('ENCODER_1', train=train)
        self.c2 = conv_block('ENCODER_2', train=train)
        self.c3 = conv_block('ENCODER_3', train=train)
        self.c4 = conv_block('ENCODER_4', train=train)
        self.c5 = conv_block('ENCODER_5', train=train)
        self.c6 = conv_block('ENCODER_6', train=train)

    def __call__(self, x):
        return self.c6(self.c5(self.c4(self.c3(self.c2(self.c1(x))))))

class io_mixer():
    def __init__(self):
        self.att    = attend('IO_MIX_attention')
        self.concat = KL.Concatenate(name='IO_MIX_concat')
        self.conv   = conv_step(100, 3, 'same', 1, name='IO_MIX')

    def __call__(self, x, y):
        return self.conv(self.concat([self.att(x, y), y]))

class decoder():
    def __init__(self):
        self.conv1 = conv_block('DECODER_conv1', 1)
        self.att1  = attend('DECODER_att_1')
        self.conv2 = conv_block('DECODER_conv2', 1)
        self.att2  = attend('DECODER_att_2')
        self.conv3 = conv_block('DECODER_conv3', 1)
        self.att3  = attend('DECODER_att_3')
        self.conv4 = conv_block('DECODER_conv4', 1)
        self.att4  = attend('DECODER_att_4')
        self.add   = KL.Add(name='DECODER_add')
        self.logit = KL.Dense(1, activation='softmax', name='DECODER_OUT')

    def __call__(self, enc, x):
        x = self.add([self.conv1(x), self.att1(enc, x)])
        x = self.add([self.conv2(x), self.att2(enc, x)])
        x = self.add([self.conv3(x), self.att3(enc, x)])
        x = self.add([self.conv4(x), self.att4(enc, x)])
        return self.logit(x)

class encoding_stage():
    def __init__(self, maxLen, vocab_size, sins, train):
        self.embed = KL.Embedding(vocab_size, 100, input_length=maxLen, name='Input_Embedding')
        self.pos = KL.Lambda(lambda x: x*sins,name='Positional_Encoding')
        self.enc = encoder(train=train)

    def __call__(self, Input):
        return self.enc(self.pos(self.embed(Input)))

class decoding_stage():
    def __init__(self, maxLen, vocab_size, sins):
        self.maxLen = maxLen
        self.embed = KL.Embedding(vocab_size, 100, input_length=maxLen, name='Output_Embedding')
        self.timing = KL.Lambda(lambda x: x*sins,name='Timing_Encoding')
        self.concat = KL.Lambda(lambda x, i: KB.concatenate([x[:,:i], x[:,i:]*0], axis=1))
        self.mix = io_mixer()
        self.dec = decoder()
        self.reshape = KL.Reshape((maxLen,))
        self.slice = KL.Lambda(lambda x, i: x[:,i:i+1])
        self.output = []

    def __call__(self, enc, Input):
        c = self.timing(self.embed(Input))
        c = KL.Lambda(lambda x: KB.concatenate([x[:,:0], x[:,0:]*0], axis=1))(c)
        mix = self.mix(enc, c)
        dec = self.dec(enc, mix)
        out = self.reshape(dec)
        output = KL.Lambda(lambda x: x[:,:1])(out)
        self.output.append(output)

        for i in range(1,self.maxLen):
            c = self.timing(self.embed(out))
            c = KL.Lambda(lambda x: KB.concatenate([x[:,:i], x[:,i:]*0], axis=1))(c)
            mix = self.mix(enc, c)
            dec = self.dec(enc, mix)
            out = self.reshape(dec)
            output = KL.Lambda(lambda x: x[:,i:i+1])(out)
            self.output.append(output)
        return KL.Lambda(lambda x: KB.stack(x, axis=1))(self.output)

class SliceNet():
    def __init__(self, vocab_size=4000, maxLen=50, depth=100):
        '''
        Creates a SliceNet Model
        '''
        self.vocab_size = vocab_size
        self.maxLen = maxLen
        self.depth = depth
        self.sins = 0.01*np.array([
            np.sin(np.array(range(maxLen))/(10000**(2*d/100)))
            if d%2 == 0 else
            np.cos(np.array(range(maxLen))/(10000**(2*d/100)))
            for d in range(100)]).T

        self.encoding = encoding_stage(self.maxLen, self.vocab_size, self.sins, train=1)
        self.decoding = decoding_stage(self.maxLen, self.vocab_size, self.sins)
        print('Model created')


    def compile(self, optimizer, loss):
        self.in1 = KL.Input(shape=(self.maxLen,), name='Input')
        self.in2 = KL.Input(shape=(self.maxLen,), name='Output')
        enc = self.encoding(self.in1)
        out = self.decoding(enc, self.in1)
        self.model = K.models.Model([self.in1, self.in2], out)
        self.model.compile(optimizer=optimizer, loss=loss)
        print('Model comiled')
        self.model.summary()

#sn = SliceNet()
#sn.compile('Adam', 'binary_crossentropy')
#sn.model.save('SliceNet.h5')
#model_json = sn.model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
#K.utils.plot_model(sn.model, to_file='model.png')

#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

#from keras.utils import plot_model
#plot_model(sn, to_file='model.png', dpi=600)

'''
dec_outputs = []

        decodings = self.o_word_emb(dec_inputs)
        # this splits the full input tensor into individual time steps that each step may be individually processed
        decodings = Lambda(lambda x: tf.unstack(x, num=self.max_dec_seq_len, axis=1))(decodings)
        for decoding in decodings:
            decoding = self.embedding_reshaper(decoding)
            context, _ = self.attention(encoding, decoding, h)
            decoding, h, c = self.decoder1(context, initial_state=[h, c])
            decoding = self.decoder_reshaper(decoding)
            decoding, h, c = self.decoder2(decoding, initial_state=[h, c])
            logits = self.output_layer(decoding)

            dec_outputs.append(logits)

        dec_outputs = Lambda(lambda x: K.stack(x, axis=1))(dec_outputs)
'''
