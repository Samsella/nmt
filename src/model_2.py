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
    ''' One step of Seperable Convolution and Normalization
        with padding for the autoregressive mode
    '''
    def __init__(self, filters, length, pad='same', dil=1 ,name=''):
        with scope('conv_step'):
            self.pad     = pad
            self.padding = KL.ZeroPadding1D(padding=((length-1)*dil,0))
            self.act     = KL.Activation('relu', name=name+'_Act')
            self.conv    = KL.SeparableConv1D(filters, length, padding=pad, dilation_rate=dil, name=name+'_SepConv')
            self.norm    = LayerNormalization(name=name+'_Norm')

    def __call__(self, x):
        with scope('conv_step_op'):
            if self.pad != 'same':
                x = self.padding(x)
                print(x)
            return self.norm(self.conv(self.act(x)))


class conv_block():
    ''' One Convolution block consisting of 4 conv_steps with residual
        connections
    '''
    def __init__(self, name, pad='same', depth=1000):
        with scope('conv_block'):
            self.conv1   = conv_step(depth, 3, pad=pad, name=name+'_1')
            self.conv2   = conv_step(1000, 3, pad=pad, name=name+'_2')
            self.add1    = KL.Add(name=name+'_Add_1')
            self.conv3   = conv_step(depth, 15, pad=pad, name=name+'_3')
            self.conv4   = conv_step(1000, 15, pad=pad, dil=4,name=name+'_4')
            self.add2    = KL.Add(name=name+'_Add_2')
            self.dropout = KL.Dropout(0.5, name=name+'_Dropout')

    def __call__(self, x, train):
        with scope('conv_block'):
            y   = self.add1([x, self.conv2(self.conv1(x))])
            out = self.add2([x, self.conv4(self.conv3(y))])
            out = self.dropout(out, training=train)
            return out

class attend():
    ''' Attention block
    '''
    def __init__(self, name, pad='same', depth=1000):
        with scope('attention'):
            self.coeff = KL.Lambda(lambda x: x/np.sqrt(depth), name=name+'_Coeff')
            self.conv1 = conv_step(depth, 5, pad=pad, dil=1, name=name+'_1')
            self.conv2 = conv_step(depth, 5, pad=pad, dil=2, name=name+'_2')
            #self.T     = KL.Lambda(lambda x: KB.permute_dimensions(x, (0,2,1)), name=name+'_Transpose')
            self.dot1  = KL.Dot(axes=(2,2), name=name+'_Dot_1')
            self.softm = KL.Softmax(name=name+'_Softmax')
            self.dot2  = KL.Dot(axes=(2,1), name=name+'_Dot_2')

    def __call__(self, x, y):
        with scope('attention_op'):
            c   = self.coeff(x)
            #x_T = self.T(x)
            y   = self.conv2(self.conv1(y))
            out = self.dot2([self.softm(self.dot1([y, x])), c])
            return out

class encoder():
    ''' Encoder consisting of 6 convolution blocks
    '''
    def __init__(self):
        with scope('encoder'):
            self.c1 = conv_block('ENCODER_1')
            self.c2 = conv_block('ENCODER_2')
            self.c3 = conv_block('ENCODER_3')
            self.c4 = conv_block('ENCODER_4')
            self.c5 = conv_block('ENCODER_5')
            self.c6 = conv_block('ENCODER_6')

    def __call__(self, x, train):
        with scope('encoder_op'):
            return self.c6(self.c5(self.c4(self.c3(self.c2(self.c1(x, train=train), train=train), train=train), train=train), train=train), train=train)

class io_mixer():
    ''' Mixer block with encoded Input and Output of the model
    '''
    def __init__(self):
        with scope('mixer'):
            self.att    = attend('IO_MIX_attention', pad='valid')
            self.concat = KL.Concatenate(name='IO_MIX_concat')
            self.conv   = conv_step(1000, 3, 'valid', 1, name='IO_MIX')

    def __call__(self, x, y):
        with scope('mixer_op'):
            return self.conv(self.concat([self.att(x, y), y]))

class decoder():
    ''' Decoder getting input from the Encoder and the output of the mixer
    '''
    def __init__(self, vocab_size):
        with scope('decoder'):
            self.conv1 = conv_block('DECODER_conv1', pad='valid')
            self.att1  = attend('DECODER_att_1', pad='valid')
            self.conv2 = conv_block('DECODER_conv2', pad='valid')
            self.att2  = attend('DECODER_att_2', pad='valid')
            self.conv3 = conv_block('DECODER_conv3', pad='valid')
            self.att3  = attend('DECODER_att_3', pad='valid')
            self.conv4 = conv_block('DECODER_conv4', pad='valid')
            self.att4  = attend('DECODER_att_4', pad='valid')
            self.add   = KL.Add(name='DECODER_add')
            #self.logit = KL.Dense(vocab_size, name='DECODER_DENSE')
            self.logit = KL.Dense(vocab_size, activation='softmax', name='DECODER_DENSE')
            #self.logit = KL.Softmax(name='DECODER_OUT')
            self.argmax = KL.Lambda(lambda x: KB.argmax(x))

    def __call__(self, enc, x, train):
        with scope('decoder_op'):
            x = self.add([self.conv1(x, train=train), self.att1(enc, x)])
            x = self.add([self.conv2(x, train=train), self.att2(enc, x)])
            x = self.add([self.conv3(x, train=train), self.att3(enc, x)])
            x = self.add([self.conv4(x, train=train), self.att4(enc, x)])
            x = self.logit(x)
            y = self.argmax(x)
            return x, y

class encoding_stage():
    ''' This is the whole Encoder with Embedding and positional encoding
    '''
    def __init__(self, maxLen, vocab_size, sins):
        with scope('encoding_stage'):
            self.embed = KL.Embedding(vocab_size, 1000, input_length=maxLen, name='Input_Embedding')
            self.pos   = KL.Lambda(lambda x: x+sins,name='Positional_Encoding')
            self.enc   = encoder()

    def __call__(self, Input, train=1):
        with scope('encoding_stage_op'):
            return self.enc(self.pos(self.embed(Input)), train=train)

class decoding_stage():
    ''' This is the whole decoder with Embedding and timing
    '''
    def __init__(self, maxLen, vocab_size, sins):
        with scope('decoding_stage'):
            self.maxLen  = maxLen
            self.embed   = KL.Embedding(vocab_size, 1000, name='Output_Embedding')
            self.timing  = KL.Lambda(lambda x: x+sins,name='Timing_Encoding')
            self.mix     = io_mixer()
            self.dec     = decoder(vocab_size)
            #self.reshape = KL.Reshape((maxLen,))
            #self.embed_reshape = KL.Reshape((100,))
            #self.unstack = KL.Lambda(lambda x: tf.unstack(x, num=maxLen, axis=1))
            #self.i = 0
            #self.slice = KL.Lambda(lambda x: x[:,self.i:self.i+1])
            #self.stack = KL.Lambda(lambda x: KB.stack(x, axis=1))
            #self.output = []

    def __call__(self, enc, Input, train):
        with scope('decoding_stage_op'):
            c        = self.timing(self.embed(Input))
            #c       = KL.Lambda(lambda x: KB.concatenate([x[:,:0], x[:,0:]*0], axis=1))(c)
            mix      = self.mix(enc, c)
            log, dec = self.dec(enc, mix, train=train)
            #out      = self.reshape(dec)
            #if self.train:
            return log, dec

class SliceNet():
    def __init__(self, vocab_size=4000, maxLen=40, depth=1000):
        '''
        Creates a SliceNet Model
        '''
        self.train = tf.placeholder_with_default(True, (), name= 'training')
        self.vocab_size = vocab_size
        self.maxLen     = maxLen
        self.depth      = depth
        self.sins       = 0.01*np.array([
            np.sin(np.array(range(maxLen))/(10000**(2*(d//2)/depth)))
            if d%2 == 0 else
            np.cos(np.array(range(maxLen))/(10000**(2*(d//2)/depth)))
            for d in range(depth)]).T

        self.encoding = encoding_stage(self.maxLen, self.vocab_size, self.sins)
        self.decoding = decoding_stage(self.maxLen, self.vocab_size, self.sins)
        print('Model created')


    def compile(self, optimizer, loss):
        self.in1   = KL.Input(shape=(self.maxLen,), name='Input')
        self.in2   = KL.Input(shape=(self.maxLen,), name='Output')
        enc        = self.encoding(self.in1, self.train)
        log, out   = self.decoding(enc, self.in2, self.train)
        self.model = K.models.Model([self.in1,self.in2], log)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])
        print('Model comiled')
        #self.model.summary()

    def predict(self, src):
        self.train = 0
        batch_size = src.shape[0]
        output = np.zeros((batch_size, self.maxLen), dtype=int)
        output[:,0] = 1
        for idx in range(self.maxLen-2):
            log = self.model.predict([src, output])
            dec_out = np.argmax(log, axis=2)
            output[:, idx+1] = dec_out[:,idx]
        return output

#sn = SliceNet()
#sn.compile('Adam', 'binary_crossentropy')
#sn.model.summary()
