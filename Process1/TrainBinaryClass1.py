########## 0 - Importing relevant libraries ##########
import h5py
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from keras import initializers
from keras import regularizers , optimizers
from sklearn.model_selection import train_test_split
import scipy.io as sio

########## 1 - Loading the data sets ##########
## loading data
data = h5py.File('/floyd/input/binary1/Binary1.mat','r')

X1 = np.array(data['Echogram1'])
label1 = np.array(data['label1'])

Y = to_categorical(label1)
Y = Y[:,1:]

## Plot the distribtion
plt.hist(label1)

########## 2 - Baseline model ########## 
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(155, input_dim=150, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(166,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(89, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))      
    model.add(Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

snn = baseline_model()
snn.summary()

## 1.2 Training the model
history = snn.fit(X1, Y, epochs = 6, batch_size = 8192, shuffle=True)
error = snn.evaluate(X1,Y,batch_size = 8192)
print('Error:',error)

# Save the Neural Network trained
snn.save('binary1.h5')
