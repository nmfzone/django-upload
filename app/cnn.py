# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import theano
from theano import tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
import numpy as np
import _pickle as cPickle
from math import sqrt

from PIL import Image
import matplotlib.pyplot as plt
from django.conf import settings


class CNNTest:
    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)  # @UndefinedVariable

    def rectify(self, X):
        return T.maximum(X, 0.)

    def softmax(self, X):
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

    def generateHeWeights(self, count, n):
        return np.random.randn(count) * sqrt(2.0/n)

    def init_conv_weights(self, shape):
        weights = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                w = self.generateHeWeights(shape[2]*shape[3], shape[1])
                w = w.reshape(shape[2], shape[3])
                weights[i][j] = w
        return theano.shared(self.floatX(weights))

    def init_xavier_weights(self, shape):
        weights = np.random.uniform((-np.sqrt(6.0 / (shape[0] + shape[1]))), (np.sqrt(6.0 / (shape[0] + shape[1]))), size=(shape[0], shape[1]))
        return theano.shared(self.floatX(weights))

    def init_weights(self, shape):
        return theano.shared(self.floatX(np.zeros(shape)))

    def init_bias(self, shape):
        return theano.shared(self.floatX(np.zeros(shape)))

    def load_weights(self, w, b, w2, b2, w3, b3, w4, b4, w_o, b_o):
        bobot = os.path.join(settings.STORAGE_ROOT, 'bobot2_CNN.save')
        save_file = open(bobot, 'rb')
        w.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)
        b.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)
        w2.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)
        b2.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)
        w3.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)
        b3.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)
        w4.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)
        b4.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)
        w_o.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)
        b_o.set_value(cPickle.load(save_file, encoding='latin1'), borrow=True)

    def model(self, X, w, b, w2, b2, w3, b3, w4, b4, w_o, b_o):
        l1 = self.rectify(conv2d(X, w) + b.dimshuffle('x', 0, 'x', 'x'))
        l1 = pool_2d(l1, (2, 2), ignore_border=True)
        l2 = self.rectify(conv2d(l1, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
        l2 = pool_2d(l2, (2, 2), ignore_border=True)
        l3 = self.rectify(conv2d(l2, w3) + b3.dimshuffle('x', 0, 'x', 'x'))
        l3 = pool_2d(l3, (2, 2), ignore_border=True)
        l3 = T.flatten(l3, outdim=2)
        l4 = T.tanh(T.dot(l3, w4) + b4)
        pyx = self.softmax(T.dot(l4, w_o) + b_o)
        return pyx

    def run(self, imagePath):
        X = T.ftensor4()

        w = self.init_conv_weights((32, 1, 3, 3))
        b = self.init_bias((32,))
        w2 = self.init_conv_weights((64, 32, 2, 2))
        b2 = self.init_bias((64,))
        w3 = self.init_conv_weights((128, 64, 3, 3))
        b3 = self.init_bias((128,))
        w4 = self.init_xavier_weights((512, 1000))
        b4 = self.init_bias((1000,))
        w_o = self.init_xavier_weights((1000, 20))
        b_o = self.init_bias((20,))

        py_x = self.model(X, w, b, w2, b2, w3, b3, w4, b4, w_o, b_o)
        y_x = T.argmax(py_x, axis=1)

        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

        self.load_weights(w, b, w2, b2, w3, b3, w4, b4, w_o, b_o)

        img = Image.open(imagePath).convert('L')
        img = img.resize((28, 28), Image.ANTIALIAS)
        img = np.array(img)
        img = 255 - img
        img = img / 255.

        print(img)

        plt.imshow(img, cmap='Greys')
        plt.show()

        img = img.reshape(1, 1, 28, 28)
        hasil = (predict(img)+1)
        print(hasil)
