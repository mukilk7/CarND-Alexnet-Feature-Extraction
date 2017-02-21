#!/usr/bin/env python

import pickle
import numpy as np
import math

import pdb

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.datasets import cifar10
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


def normalizeData(dat, a = -0.5, b = 0.5, xmin = 0, xmax = 255):
    normed = a + ( ((dat - xmin) * (b - a)) / (xmax - xmin) )
    return normed


def getModel(feature_extract = False):
    model = Sequential()
    # apply a 3x3 convolution with 6 output filters on a 32x32x3 image:
    model.add(Convolution2D(6, 3, 3, border_mode='valid', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    model.add(Dropout(0.25))
    model.add(Activation("relu"))
    # apply another 3x3 convolution with 16 output filters to 15x15x6 input:
    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    #input_shape = 6x6x16, here
    model.add(Flatten())
    #input_shape = 576, fully connected 1
    model.add(Dense(120))
    model.add(Activation("relu"))
    #fully connected 2
    model.add(Dense(84))
    model.add(Activation("relu"))
    #optional final fully connected layer
    if not feature_extract:
        model.add(Dense(43))
        model.add(Activation("softmax"))
    return model


def getInputData(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    X_data = data['features']
    y_data = data['labels']
    X_data, y_data = shuffle(X_data, y_data)
    print("Starting norming")
    nfunc = np.vectorize(normalizeData)
    X_normalized = nfunc(X_data)
    print(X_normalized.shape)
    print("Done norming")
    lb = LabelBinarizer()
    y_one_hot = lb.fit_transform(y_data)
    y_one_hot = y_one_hot.astype(np.float32)
    return X_normalized, y_one_hot


def trainModel(model, X_normalized, y_one_hot, epochs = 10, batchsz=128, valsplit=0.05):
    model.compile('sgd', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_normalized, y_one_hot, batch_size=batchsz, nb_epoch=epochs, validation_split=valsplit)


def evaluateModelOnTest(filename, model):
    X_test, one_hot_y_test = getInputData(filename)
    metrics = model.evaluate(X_test, one_hot_y_test)
    return metrics


def printMetrics(model, metrics):
    print("\n=== Metrics ===\n")
    for mi in range(len(model.metrics_names)):
        mname = model.metrics_names[mi]
        mval = metrics[mi]
        print("{}: {}".format(mname, mval))


def GTSRB_Run():
    X_normalized, y_one_hot = getInputData("train.p")
    pdb.set_trace()
    model = getModel()
    trainModel(model, X_normalized, y_one_hot)
    metrics = evaluateModelOnTest("test.p", model)
    printMetrics(model, metrics)

    
def CIFAR10_Run():
    n_cifar10_classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    nfunc = np.vectorize(normalizeData)
    print("Starting norming...")
    X_train = nfunc(X_train)
    y_train = y_train.reshape(-1)
    X_train, y_train = shuffle(X_train, y_train)
    X_test = nfunc(X_test)
    y_test = y_test.reshape(-1)
    print("Done norming")
    model = getModel(feature_extract = True)
    #add new final layer to reflect cifar10 classes
    model.add(Dense(n_cifar10_classes))
    model.add(Activation("softmax"))
    #TODO: one_hot_y?
    lb = LabelBinarizer()
    one_hot_y_train = lb.fit_transform(y_train)
    trainModel(model, X_train, one_hot_y_train)
    one_hot_y_test = lb.fit_transform(y_test)
    metrics = model.evaluate(X_test, one_hot_y_test)
    printMetrics(model, metrics)    
        
###

#GTSRB_Run()
CIFAR10_Run()
