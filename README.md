# NEUWORK

## INTRODUCTION
Neuwork is a module that allows you to create any multilayer perceptron machines.

## INSTALLATION
To work with this module just copy the 'NN_module' folder and paste it to your project directory.

## APPLICATION
### 1. Importing module calsses
In order to build a neural network first import the required files 
<br>
<code>
from NN_modeules.neural_network import NN
</code>
<br>
this will import the NN module which builds the neural network
<br>
<code>
from NN_modeules.layers import Dense
</code>
<br>
to import the type of layer you want to work with

### 2. Initializing NN
for buildinbg the neural network first we need to build layers of multiple perceptrons which will be used by the neural network.\
#### Layer creation : 
<code>
layer = Dense(units = ?, activation = ?, weights_regularizer_l1 = ?, weights_regularizer_l2 = ?, bias_regularizer_l1 = ?, bias_regularizer_l2 = ? )
</code>
<br>

1. **units** : number of perceptron/neuron in the layer (Eg : 10,21, 69,....)
2. **activation** : String that denotes the type of activation the layer will have. there are few activations available wich are {**'linear', 'relu', 'softmax', 'sigmoid'**}. **The default value of the activation parameter is 'relu'**
3. **weights_regularizer_l1** : lambda value for L1 weight regularization (default value is 0)
4. **weights_regularizer_l2** : lambda value for L2 weight regularization (default value is 0)
5. **bias_regularizer_l1** : lambda value for L1 bias regularization (default value is 0)
6. **bias_regularizer_l2** : lambda value for L2 bias regularization (default value is 0)

Example : 
<code>
layer1 = Dense(units = 10, activation = 'softmax')
layer2 = Dense(units = 128, activation = 'relu', weights_regularizer_l2 = 0.04)
</code>
<br>

#### Neural Network Creation : 
