! pip install gpyopt

import h5py
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import GPyOpt

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
    model.add(Dense(150, input_dim=150, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(75,kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(25, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))      
    model.add(Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid'))
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

snn = baseline_model()
snn.summary()

## Running the model on one epoch to get a baseline score to minimize
history = snn.fit(X1, Y, epochs = 1, batch_size = 8192, shuffle=True)
baseline_score = history.history["loss"][0]

########## 3 - Bayesian Optimisation ##########
# Model to optimize
# We are going to allow the learning rate, and the number of neuron in each layer 
def variable_neural_network_model(learning_rate,no_unit_layer1,no_unit_layer2,no_unit_layer3):
    # create model
    model = Sequential()
    model.add(Dense(no_unit_layer1, input_dim=150, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(no_unit_layer2, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(no_unit_layer3, kernel_initializer="normal", bias_initializer='zeros', activation='relu'))  
    model.add(Dense(2, kernel_initializer="normal", bias_initializer='zeros', activation='sigmoid'))
    # Custom optimizer
    adam = optimizers.Adam(lr=learning_rate)
	# Compile model
    model.compile(loss='binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    return model


# Function to optimize
# We aim to minimize this function with respect to the parameters
# learning_rate and number of neurons in each layer
def f(x):
    learning_rate = float(x[:,0])
    no_unit_layer1 = int(x[:,1])
    no_unit_layer2 = int(x[:,2])
    no_unit_layer3 = int(x[:,3])     
    vnn = variable_neural_network_model(learning_rate,no_unit_layer1,no_unit_layer2,no_unit_layer3)
    history = vnn.fit(X1, Y,epochs = 1, batch_size = 1024, shuffle=True)
    score = history.history["loss"][0]
    return score

## Here is the bound for the hyperparameters to vary
bounds = [
            {'name': 'learning_rate', 'type': 'continuous', 'domain': (10**-4,10**-2)},
            {'name': 'no_unit_layer1', 'type': 'discrete', 'domain': np.arange(5,200)},
            {'name': 'no_unit_layer2', 'type': 'discrete', 'domain': np.arange(5,200)},
            {'name': 'no_unit_layer3', 'type': 'discrete', 'domain': np.arange(5,200)},
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

# We then measure how much the performance has progress in percentage and
# show it on the screen
performance_boost_Binary1 = (baseline_score/np.min(optimizer.Y) -1)*100
print('Performance improvement >',np.floor(performance_boost_Binary1),'%')


## Finaly we save our hyperparameters and show them on the screen
best_learning_rate1, first_Layer1, second_Layer1, third_Layer1 = optimizer.X[np.argmin(optimizer.Y)]
print('Best_lerning_rate: ',best_learning_rate1)
print('first_Layer1: ',first_Layer1)
