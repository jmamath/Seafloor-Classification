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


### Defining the Importance Weighted Empirical Risk Minimizer iwERM
def iwERM(importance):
    def loss(y_true, y_pred):
         return K.mean(importance * K.binary_crossentropy( y_true, y_pred ))
    return loss


def Importance_variable_CNN(learning_rate,regularizer, kernel_1, kernel_2, kernel_3, no_unit1, no_unit2, no_unit3):
    ## Input Layer
    inputs = Input(shape=(2550,1))
    importance = Input(shape=(1,))
    ## 1st Conv Layer
    x = Conv1D(filters=1, kernel_size=kernel_1, strides=2, activation='relu')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(pool_size=kernel_1, strides=1, padding='valid')(x)
    
    ## 2nd Conv Layer
    x = Conv1D(filters=1, kernel_size=kernel_2, strides=1, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(pool_size=kernel_2, strides=1, padding='valid')(x) 
    
    ## 3rd Conv Layer
    x = Conv1D(filters=1, kernel_size=kernel_3, strides=1, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling1D(pool_size=kernel_3, strides=1, padding='valid')(x) 
    x = Flatten()(x)
    
    ## Fully Connected Layers
    x = Dense(no_unit1,kernel_initializer="normal", bias_initializer='zeros', activation='relu')(x)
    x = Dense(no_unit2,kernel_initializer="normal", bias_initializer='zeros', activation='relu')(x)
    x = Dense(no_unit3,kernel_initializer="normal", bias_initializer='zeros', activation='relu')(x)
    predictions = Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid', kernel_regularizer=regularizers.l2(regularizer))(x)
    
    # Custom optimizer
    adam = optimizers.Adam(lr=learning_rate)
    ## Getting the model
    model = Model(inputs=[inputs,importance], outputs = predictions)
    model.compile(optimizer=adam, loss=iwERM(importance), metrics=['accuracy'])
    return model

# Loading the hyperparameters into our model
vnn = Importance_variable_CNN(1.03538939e-02, 7.89356334e-01, 6,  15,  31, 106, 170,  67)

## Training the model and evaluating it on the validation data
history = vnn.fit([X_train,importance], Y_train, validation_data = ([X_test,importance_bis],Y_test), epochs = 6, batch_size = 1024)


# Save the Neural Network trained
vnn.save_weights('AdaptatedWeights_binary2.h5')

