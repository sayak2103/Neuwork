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
    def init_layer(self, n_1) :
        self.W = 0.01*np.random.randn(n_1, self.n)
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
        dvalues = self.g.get_g_grad(self.outputs , grad)
        #dvalues = q * grad
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
            self.db += dL1
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
#
class RNN :
    def __init__ (self, units=1, activation='tanh',
                weights_regularizer_l1=0, weights_regularizer_l2=0,
                bias_regularizer_l1=0, bias_regularizer_l2=0) :
        self.n=units
        self.g=Activation().get_activation(activation)
        self.lambda_W1 = weights_regularizer_l1
        self.lambda_W2 = weights_regularizer_l2
        self.lambda_b1 = bias_regularizer_l1
        self.lambda_b2 = bias_regularizer_l2
        #maintaining a cache for previous time step
        self.caches = []
    #
    def set_idx(self, i) :
        self.idx = i
    #
    def init_layer(self, n_1) :
        self.Waa = 0.01*np.random.randn(self.n, self.n)
        self.Wax = 0.01*np.random.randn(n_1, self.n)
        self.ba = np.zeros((1, self.n))
        self.dWaa = np.zeros_like(self.Waa)
        self.dWax = np.zeros_like(self.Wax)
        self.dba = np.zeros_like(self.ba)
    #
    def forward_propagation(self , X , nn , time_step=0) :
        #X.shape = (m,n_1)
        a_i1_t = X
        a_i_t1 = self.caches[time_step-1] if time_step > 0 else np.zeros((X.shape[0], self.n))
        # feedforward
        a_i_t = self.g.activate(a_i1_t @ self.Wax + a_i_t1 @ self.Waa + self.ba)
        cache = (a_i1_t , a_i_t1 , a_i_t)
        self.caches.append(cache)
        #returning the output of the last layer
        if(self.idx < nn.num_layers-1) :
            return nn.layers[self.idx + 1].forward_propagation(a_i_t, nn , time_step)
        else :
            return a_i_t
    #
    # function for backpropagation
    def backward_propagation(self , grad , nn , time_step) :
        # grad.shape = (m,n)
        # grad is the gradient of the loss function or the upper layer \/
        # we need the gradient from this current layer of next time step <--
        if(time_step < len(self.caches)-1) :
            grad += self.caches[time_step+1][3]
        # grad = dL/da_i_t
        #output of the current layer 
        a_i_t = self.caches[time_step][2] #(m,n)
        #input of this layer
        a_i1_t = self.caches[time_step][0] #(m,n_1) from previous layer
        a_i_t1 = self.caches[time_step][1] #(m,n) from previous time step
        #self.Wax (n_1,n)
        #self.Waa (n,n)
        #self.ba (1,n)
        #
        grad = self.g.get_g_grad(a_i_t , grad) #(m,n)
        # grad = dL/dz_i_t
        dWax = a_i1_t.T @ grad #(n_1,n)
        dWaa = a_i_t1.T @ grad #(n,n)
        db = np.sum(grad, axis=0, keepdims=True) / grad.shape[0] #(1,n)
        self.dWaa += dWaa
        self.dWax += dWax
        self.dba += db
        #
        da_i1_t = grad @ self.Wax.T #(m,n_1)
        da_i_t1 = grad @ self.Waa.T #(m,n)
        self.caches[time_step].append(da_i_t1)
        if(self.idx > 0) :
            nn.layers[self.idx - 1].backward_propagation(da_i1_t , nn , time_step)
    #
#
    # def regularization_penalty(self) :
    #     cost = 0
    #     if(self.lambda_W1 > 0) :
    #         cost += self.lambda_W1 * np.sum(np.abs(self.Waa))
    #         cost += self.lambda_W1 * np.sum(np.abs(self.Wax))
    #     if(self.lambda_b1 > 0) :
    #         cost += self.lambda_b1 * np.sum(np.abs(self.ba))
    #     if(self.lambda_W2 > 0) :
    #         cost += self.lambda_W2 * np.sum((self.Waa)**2)
    #         cost += self.lambda_W2 * np.sum((self.Wax)**2)
    #     if(self.lambda_b2 > 0) :
    #         cost += self.lambda_b2 * np.sum((self.ba)**2)
    #     return cost
    #