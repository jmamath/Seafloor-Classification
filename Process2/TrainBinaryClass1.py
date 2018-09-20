#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:12:55 2018

@author: root
"""
########## 0 - Importing relevant libraries ##########

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio

import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from keras import regularizers , optimizers

from keras.utils import to_categorical
from keras.utils import plot_model


########## 1 - Loading the data sets ##########
##### 1.1 loading the training data

data = h5py.File('/floyd/input/binary1_1/Binary1_1.mat','r')
X = np.array(data['Echogram1_1'])
label = np.array(data['label1_1'])

Y = to_categorical(label)
Y = Y[:,1:]

##### 1.2 Loading validation data

test = sio.loadmat('/floyd/input/sample1_1/Sample1_1.mat')
X_test = test['Echogram']
label_test = test['label'] 

Y_test = to_categorical(label_test)
Y_test = Y_test[:,1:]

##### 1.3 - Load importance weights
data3 = sio.loadmat('Importance.mat')
importance = data3['importance']
importance_bis = np.ones(m).reshape(m,1)
plt.hist(importance)


########## 2 - Setting up the model 

### Defining the Importance Weighted Empirical Risk Minimizer iwERM
def iwERM(importance):
    def loss(y_true, y_pred):
         return K.mean(importance * K.binary_crossentropy( y_true, y_pred ))
    return loss


########## 1.2 - Model without regularization
def Importance_variable_CNN(kernel_1, kernel_2, kernel_3, no_unit1, no_unit2):
    ## Input Layer
    inputs = Input(shape=(2550,1))
    importance = Input(shape=(1,))
    ## 1st Conv Layer
    x = Conv1D(filters=1, kernel_size=kernel_1, strides=2, activation='relu')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(pool_size=kernel_1, strides=2, padding='valid')(x)
    
    ## 2nd Conv Layer
    x = Conv1D(filters=1, kernel_size=kernel_2, strides=2, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(pool_size=kernel_2, strides=2, padding='valid')(x) 
    
    ## 3rd Conv Layer
    x = Conv1D(filters=1, kernel_size=kernel_3, strides=1, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(pool_size=kernel_3, strides=1, padding='valid')(x) 
    x = Flatten()(x)
    ## Fully Connected Layers
    x = Dense(no_unit1,kernel_initializer="normal", bias_initializer='zeros', activation='relu')(x)
    x = Dense(no_unit2,kernel_initializer="normal", bias_initializer='zeros', activation='relu')(x)
    predictions = Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid')(x)
    
    ## Getting the model
    model = Model(inputs=[inputs,importance], outputs = predictions)
    model.compile(optimizer='adam', loss=iwERM(importance), metrics=['accuracy'])
    return model

vnn = Importance_variable_CNN(42, 13, 45, 60, 72)
vnn.summary()

## Training the model and evaluating it on the validation data
history = vnn.fit([X,importance], Y, validation_data = ([X_test,importance_bis],Y_test), epochs = 36, batch_size = 1024)



# Save the Neural Network trained
vnn.save_weights('AdaptatedWeights_binary1.h5')