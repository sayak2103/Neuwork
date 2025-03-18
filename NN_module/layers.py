#This python file contains the calss for layers in each neural network
import numpy as np
from .activation_func import Activation
#from .neural_network import NN
class Dense :
    #constructor to pass <no. of perceptron in this layer / activation function >
    def __init__(self, units=1, activation='relu',
                weights_regularizer_l1=0, weights_regularizer_l2=0,
                bias_regularizer_l1=0, bias_regularizer_l2=0) :
        self.n=units
        self.g=Activation().get_activation(activation)
        self.lambda_W1 = weights_regularizer_l1
        self.lambda_W2 = weights_regularizer_l2
        self.lambda_b1 = bias_regularizer_l1
        self.lambda_b2 = bias_regularizer_l2
        

    def set_idx(self, i) :
        self.idx = i
    #
    def init_layer(self, m) :
        self.W = 0.01*np.random.randn(m, self.n)
        self.b = np.zeros((1, self.n))
        self.W_momentum = np.zeros_like(self.W)
        self.b_momentum = np.zeros_like(self.b)
    #
    #def set_layer_momentum(self, m) :
    #    self.layer_momentum = m
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
    def regularization_penalty(self) :
        cost = 0
        if(self.lambda_W1 > 0) :
            cost += self.lambda_W1 * np.sum(np.abs(self.W))
        if(self.lambda_b1 > 0) :
            cost += self.lambda_b1 * np.sum(np.abs(self.b))
        if(self.lambda_W2 > 0) :
            cost += self.lambda_W2 * np.sum((self.W)**2)
        if(self.lambda_W1 > 0) :
            cost += self.lambda_b2 * np.sum((self.b)**2)
        return cost
    
    #
    def backward_propagation(self, grad, nn) :
        # grad.dimension = m * self.n
        q = self.g.get_g_grad(self.outputs)
        dvalues = q * grad
        self.dW = np.matmul(self.inputs.T, dvalues)
        self.db = np.sum(grad, axis=0, keepdims=True)
        dinputs = np.matmul(dvalues, self.W.T)
        #adjusting the gradients for regularization
        if self.lambda_W1 > 0 :
            dL1 = np.ones_like(self.W)
            dL1[self.W < 0] = -1
            dL1 *= self.lambda_W1
            self.dW += dL1
        if self.lambda_b1 > 0 :
            dL1 = np.ones_like(self.b)
            dL1[self.b < 0] = -1
            dL1 *= self.lambda_b1
            self.db += temp
        if self.lambda_W2 > 0 :
            dL2 = 2*self.lambda_W2*self.W
            self.dW +=dL2
        if self.lambda_b2 > 0 :
            dL2 = 2*self.lambda_b2*self.b
            self.db +=dL2
        
        #next_grad /= self.n #this is a mistake don't uncomment, just think why it's wrong ;)
        #if hasattr(self, 'layer_momentum') :
        #   self.update_parameters(nn.l_rate, self.layer_momentum)
        #else :
        #    self.update_parameters(nn.l_rate, nn.momentum)
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
        
            
        