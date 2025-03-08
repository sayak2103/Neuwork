#This python file contains the calss for layers in each neural network
import numpy as np
from .activation_func import Activation
#from .neural_network import NN
class Dense :
    #constructor to pass <no. of perceptron in this layer / activation function >
    def __init__(self, units=1, activation='relu') :
        self.n=units
        self.g=Activation().get_activation(activation)

    def set_idx(self, i) :
        self.idx = i
    #
    def init_layer(self, m) :
        self.W = 0.01*np.random.randn(m, self.n)
        self.b = np.zeros((1, self.n))
        self.W_momentum = np.zeros_like(self.W)
        self.b_momentum = np.zeros_like(self.b)
    #
    def set_layer_momentum(self, m) :
        self.layer_momentum = m
    #
    #function for forward propagation
    def forward_propagation(self, X, nn) :
        self.inputs = X
        Z = np.matmul(X, self.W) + self.b;
        self.outputs=self.g.activate(Z)
        if(self.idx < nn.num_layers-1) :
            return nn.layers[self.idx + 1].forward_propagation(self.outputs, nn)
        else :
            return self.outputs
    #
    def backward_propagation(self, grad, nn) :
        # grad.dimension = m * self.n
        q = self.g.get_g_grad(self.outputs)
        dvalues = q * grad
        self.dW = np.matmul(self.inputs.T, dvalues)
        self.db = np.sum(grad, axis=0, keepdims=True)
        dinputs = np.matmul(dvalues, self.W.T)
        #next_grad /= self.n
        if hasattr(self, 'layer_momentum') :
            self.update_parameters(nn.l_rate, self.layer_momentum)
        else :
            self.update_parameters(nn.l_rate, nn.momentum)
        if(self.idx > 0) :
            nn.layers[self.idx - 1].backward_propagation(dinputs, nn)
        #

    def update_parameters(self, learning_rate, momentum) :
        W_updates = momentum * self.W_momentum - learning_rate * self.dW
        b_updates = momentum * self.b_momentum - learning_rate * self.db
        self.W_momentum = W_updates
        self.b_momentum = b_updates
        self.W = self.W + W_updates
        self.b = self.b + b_updates
        
            
        