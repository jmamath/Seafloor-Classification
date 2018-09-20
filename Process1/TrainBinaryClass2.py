########## 0 - Importing relevant libraries ##########
import h5py
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from keras import initializers
from keras import regularizers , optimizers
from sklearn.model_selection import train_test_split
import scipy.io as sio

########## 1.1 - Loading the data sets ##########
## loading data
data = h5py.File('/floyd/input/echogram_raw/Second_Classification_2011.mat','r')

X = np.array(data['Echogram_Bottom'])
label = np.array(data['label_Bottom']) -1

## New distribution of classes

plt.hist(label)

Y = to_categorical(label)
Y = Y[:,1:]

########## 2 - Baseline model ########## 
best_learning_rate = 0.00072689498718629913
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(155, input_dim=150, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(166,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(89, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid'))
    # Custom optimizer
    adam = optimizers.Adam(lr=best_learning_rate)
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    return model

snn = baseline_model()
snn.summary()

## Running the model
history = snn.fit(X, Y, epochs = 400, batch_size = 8192, shuffle=True)

## Saving the trained model
snn.save('binary2.h5')
