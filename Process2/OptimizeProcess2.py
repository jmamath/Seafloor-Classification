! pip install gpyopt

import h5py
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import GPyOpt
import GPy

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

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

########## 1 - Loading the data sets ##########
##### 1.1 loading the training data

data = h5py.File('/floyd/input/binary1_1/Binary1_1.mat','r')
X = np.array(data['Echogram1_1'])
X = X[:,31:]
label = np.array(data['label1_1'])

Y = to_categorical(label)
Y = Y[:,1:]

## Plot the distribtion
plt.hist(label)

##### 1.2 Loading validation data

test = sio.loadmat('/floyd/input/sample1_1/Sample1_1.mat')
X_test = test['Echogram']
label_test = test['label'] 

Y_test = to_categorical(label_test)
Y_test = Y_test[:,1:]

## Plot data distribution
plt.hist(label_test)

# Now we want to optimize our model architecture 


########## 2 Baseline model ########## 
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(155, input_dim=2550, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(166,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(89, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))      
    model.add(Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

snn = baseline_model()
snn.summary()

## Running the model on one epoch to get a baseline score to increase
history = snn.fit(X, Y, epochs = 1, batch_size = 1024, shuffle=True)
baseline_score = history.history["loss"][0]

########## 3 - Bayesian Optimisation ##########
#####  3.1 - Optimize the number of neurons

## After some test it appears than our capacity was not enough, but a too big network
# is bad, we don't want to overfit, so we choose to use our first 3 layers
# as convolutional layers to reduce the dimensionality of the data. The last hidden
# layers are fully connected layers.

# Shaping the data
m,n = X.shape
X = X.reshape(m,n,1)

# Model to optimize
def variable_CNN(kernel_1, kernel_2, kernel_3, no_unit1, no_unit2):
    # create model
    model = Sequential()
    ## 1st Conv Layer
    model.add(Conv1D(input_shape=[2550,1], filters=1, kernel_size=kernel_1, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=kernel_1, strides=2, padding='valid'))  
    ## 2nd Conv Layer
    model.add(Conv1D(filters=1, kernel_size=kernel_2, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=kernel_2, strides=2, padding='valid')) 
    ## 3rd Conv Layer
    model.add(Conv1D(filters=1, kernel_size=kernel_3, strides=1, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=kernel_3, strides=1, padding='valid')) 
    model.add(Flatten())
    model.add(Dense(no_unit1,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(no_unit2,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid')) 
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model    


# Function to optimize
# We aim to minimize this function with respect to the number of neurons
# in the kernel of the first three hidden layers and the number of neurons in
# the two last fully connected layers.    
def f(x):
    kernel_1 = int(x[:,0])
    kernel_2 = int(x[:,1])
    kernel_3 = int(x[:,2])
    no_unit1 = int(x[:,3])
    no_unit2 = int(x[:,4])
    vnn = variable_CNN(kernel_1, kernel_2, kernel_3, no_unit1, no_unit2)
    history = vnn.fit(X, Y,epochs = 1, batch_size = 1024, shuffle=False)
    score = history.history["loss"][0]
    return score

## Here is the bound for the hyperparameters to vary
bounds = [
            {'name': 'kernel_1', 'type': 'discrete', 'domain': np.arange(5,50)},        
            {'name': 'kernel_2', 'type': 'discrete', 'domain': np.arange(5,50)},
            {'name': 'kernel_3', 'type': 'discrete', 'domain': np.arange(5,50)},        
            {'name': 'no_unit1', 'type': 'discrete', 'domain': np.arange(5,100)},
            {'name': 'no_unit2', 'type': 'discrete', 'domain': np.arange(5,100)}            
         ]

# Now we set up the Bayesian optimization procedure, the model type is a 
# Gaussian Process, and the acquisition parameter control the procedure
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds,
                                                        acquisition_type = 'EI',
                                                        acquisition_par = 0.08,                                                    
                                                        model_type='GP')

# We run it for 15 iterations
optimizer.run_optimization(max_iter=15, eps=-1)

## Plot the optimizer convergence
optimizer.plot_convergence()

# Measuring the performance improvement
performance_boost_Binary1 = (baseline_score/np.min(optimizer.Y) -1)*100
print('Performance improvement >',np.floor(performance_boost_Binary1),'%')

## Finaly we save our hyperparameters and show them on the screen
kernel_1, kernel_2, kernel_3, no_unit1, no_unit2 = optimizer.X[np.argmin(optimizer.Y)]
print('kernel_1, kernel_2, kernel_3, no_unit1, no_unit2: ', kernel_1, kernel_2, kernel_3, no_unit1, no_unit2)


#####  3.2 - Optimize the regularizer

#  Model without regularization
def variable_CNN(kernel_1, kernel_2, kernel_3, no_unit1, no_unit2):
    # create model
    model = Sequential()
    ## 1st Conv Layer
    model.add(Conv1D(input_shape=[2550,1], filters=1, kernel_size=kernel_1, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=kernel_1, strides=2, padding='valid'))  
    ## 2nd Conv Layer
    model.add(Conv1D(filters=1, kernel_size=kernel_2, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=kernel_2, strides=2, padding='valid')) 
    ## 3rd Conv Layer
    model.add(Conv1D(filters=1, kernel_size=kernel_3, strides=1, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=kernel_3, strides=1, padding='valid')) 
    model.add(Flatten())
    model.add(Dense(no_unit1,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(no_unit2,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid',kernel_regularizer=regularizers.l2(0.8))) 
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model    


vnn = variable_CNN(42, 13, 45, 60, 72)
vnn.summary()

## Running the model on one epoch to get a baseline score to increase
history = vnn.fit(X, Y, epochs = 1, validation_data=(X_test,Y_test), batch_size = 1024, shuffle=True)
baseline_score = np.min(np.array(history.history["loss"]) +  np.array(history.history["val_loss"]))


## Now we just optimize the regularization parameter

# Model to optimize
def variable_CNN(regularizer ,kernel_1, kernel_2, kernel_3, no_unit1, no_unit2):
    # create model
    model = Sequential()
    ## 1st Conv Layer
    model.add(Conv1D(input_shape=[2550,1], filters=1, kernel_size=kernel_1, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=kernel_1, strides=2, padding='valid'))  
    ## 2nd Conv Layer
    model.add(Conv1D(filters=1, kernel_size=kernel_2, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=kernel_2, strides=2, padding='valid')) 
    ## 3rd Conv Layer
    model.add(Conv1D(filters=1, kernel_size=kernel_3, strides=1, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=kernel_3, strides=1, padding='valid')) 
    model.add(Flatten())
    model.add(Dense(no_unit1,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(no_unit2,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))
    model.add(Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid',kernel_regularizer=regularizers.l2(regularizer))) 
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model    


# Function to optimize
# We aim to minimize this function with respect to the parameters
# regularizer
def f(x):
    regularizer = int(x[:,0])
    vnn = variable_CNN(regularizer ,kernel_1 = 42, kernel_2 = 13, kernel_3 = 45, no_unit1 = 60, no_unit2 = 72)
    history = vnn.fit(X, Y,epochs = 1, validation_data=(X_test,Y_test), batch_size = 1024, shuffle=True)
    loss = history.history["loss"][0]
    val_loss = history.history["val_loss"][0]
    score = loss + val_loss
    return score

## Here is the bound for the hyperparameters to vary
bounds = [
            {'name': 'regularizer', 'type': 'discrete', 'domain': np.arange(0.5,1,0.01)}                  
         ]

# Now we set up the Bayesian optimization procedure, the model type is a 
# Gaussian Process, and the acquisition parameter control the procedure
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds,
                                                        acquisition_type = 'EI',
                                                        acquisition_par = 0.1,                                                    
                                                        model_type='GP')
# We run it for 15 iterations
optimizer.run_optimization(max_iter=10, eps=-1)

## Plot the optimizer convergence
optimizer.plot_convergence()
# Measuring the performance improvement
performance_boost_Binary1 = (baseline_score/np.min(optimizer.Y) -1)*100
best_regularizer = optimizer.X[np.argmin(optimizer.Y)]
print("best_regularizer", best_regularizer)
print('Performance improvement >',np.floor(performance_boost_Binary1),'%')
