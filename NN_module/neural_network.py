import numpy as np
#from layers import * 
from .losses import Losses
from .optimizer import *
class NN : 
    #
    #contructor to initializde the neural network 
    #takes a list of layer objects as a parameter
    def __init__(self, layers, loss='mean_squared_error') : 
        self.num_layers=len(layers)
        self.layers=layers
        self.J=Losses().get_loss(loss)
        self.set_layer_idx()
        self.init_NN_layer()
        
    def set_layer_idx(self) :
        for i in range(self.num_layers) :
            self.layers[i].set_idx(i)
    #
    def init_NN_layer(self) :
        for i in range(1,self.num_layers) :
            self.layers[i].init_layer(self.layers[i-1].n)

    #function for prediction or simply froward propagation
    def predict(self, X) : 
        y_cap=self.layers[0].forward_propagation(X, self)
        return y_cap
    #
    def regularization_cost(self) :
        cost = 0
        for i in range(self.num_layers) :
            cost += self.layers[i].regularization_penalty()
        return cost
    #
    def fit(self, X, Y, epochs=100, optimizer = 'adam', learning_rate = 0.001,decay = 0.0001, 
            momentum = 0, epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.99, get_log=False) :
        if(optimizer == 'adam') : 
            self.opt = Adam(learning_rate, decay, epsilon, beta_1, beta_2)
        elif optimizer == 'sdg' :
            self.opt = SDG(learning_rate, decay, momentum)
        
        self.m = X.shape[0]
        n = X.shape[1]
        self.layers[0].init_layer(n)
        self.l_rate=learning_rate
        self.l_rate_decay=decay
        self.momentum = momentum
        #
        #now the neural network working is satrtded that combines everythig
        for i in range(epochs) :
            #self decaying learning rate, to aggresively learn at the begining, and slow down with time 
            self.l_rate = self.l_rate * (1. / (1 + (self.l_rate_decay*i)))
            #forward propagation with the inputs
            y_cap=self.predict(X)
            #computation of the cost generated
            cost=self.J.compute_cost(y_cap, Y) + self.regularization_cost()
            if(get_log and i%(epochs/10)==0) :
                print(f'epoch: {i}' + f'loss: {cost:.3f}'+'\n')
            grad=self.J.get_grad(y_cap,Y) # m*nk m =no. of sample  / nk = units in last layer
            self.layers[self.num_layers-1].backward_propagation(grad, self)
            self.opt.pre_update_params()
            for i in range(self.num_layers) :
                self.opt.update_parameters(self.layers[i])
            self.opt.post_update_params()
            