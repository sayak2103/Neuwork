#this class contains the set of various optimizers
import numpy as np

#Stochastic gradient descent
class SDG :
    #
    def __init__(self, learning_rate = 1.0, decay = 0.1, momentum = 0.) : 
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.epoch = 0
    #
    def pre_update_params(self) :
        if(self.decay > 0) :
            self.current_learning_rate = self.initial_learning_rate * (1 / (1 + self.decay * self.epoch))
    #
    def update_parameters(self, layer) :
        if(type(layer).__name__ == 'Dense') :
            if self.momentum > 0 :
                if not hasattr(layer, 'W_momentum') :
                    layer.W_momentum = np.zeros_like(layer.W)
                    layer.b_momentum = np.zeros_like(layer.b)
                #
                W_updates = self.momentum * layer.W_momentum - self.current_learning_rate * layer.dW
                b_updates = self.momentum * layer.b_momentum - self.current_learning_rate * layer.db
                layer.W_momentum = W_updates
                layer.b_momentum = b_updates
            else : 
                W_updates =  -self.current_learning_rate * layer.dW
                b_updates =  -self.current_learning_rate * layer.db
            #
            layer.W = layer.W + W_updates
            layer.b = layer.b + b_updates
        elif(type(layer).__name__ == 'RNN') :
            if self.momentum > 0 :
                if not hasattr(layer, 'Waa_momentum') :
                    layer.Waa_momentum = np.zeros_like(layer.Waa)
                    layer.Wax_momentum = np.zeros_like(layer.Wax)
                    layer.ba_momentum = np.zeros_like(layer.ba)
                #
                Waa_updates = self.momentum * layer.Waa_momentum - self.current_learning_rate * layer.dWaa
                Wax_updates = self.momentum * layer.Wax_momentum - self.current_learning_rate * layer.dWax
                ba_updates = self.momentum * layer.ba_momentum - self.current_learning_rate * layer.dba
                #
                layer.Waa_momentum = Waa_updates
                layer.Wax_momentum = Wax_updates
                layer.ba_momentum = ba_updates
            else : 
                Waa_updates =  -self.current_learning_rate * layer.dWaa
                Wax_updates =  -self.current_learning_rate * layer.dWax
                ba_updates =  -self.current_learning_rate * layer.dba
            #
            layer.Waa += Waa_updates
            layer.Wax += Wax_updates
            layer.ba += ba_updates
    #
    def post_update_params(self) :
        self.epoch += 1
    #
    #
#Adaptive momentum
class Adam : 
    #
    def __init__(self, learning_rate = 0.001, decay = 0.0001, epsilon = 1e-7, 
                beta_1 = 0.9, beta_2 = 0.999) :
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epoch = 0
    #
    def pre_update_params(self) :
        if(self.decay > 0) :
            self.current_learning_rate = self.initial_learning_rate * (1 / (1 + self.decay * self.epoch))
    #
    def update_parameters(self, layer) :
        if(type(layer).__name__ == 'Dense') :
            if not hasattr(layer, 'W_cache') :
                layer.W_momentum = np.zeros_like(layer.W)
                layer.b_momentum = np.zeros_like(layer.b)
                layer.W_cache = np.zeros_like(layer.W)
                layer.b_cache = np.zeros_like(layer.b)
            #
            layer.W_momentum = (self.beta_1 * layer.W_momentum) + ((1 - self.beta_1) * layer.dW)
            layer.b_momentum = (self.beta_1 * layer.b_momentum) + ((1 - self.beta_1) * layer.db)
            #
            W_momentum_correct = layer.W_momentum / (1 - self.beta_1 ** (self.epoch + 1))
            b_momentum_correct = layer.b_momentum / (1 - self.beta_1 ** (self.epoch + 1))
            #
            layer.W_cache = (self.beta_2 * layer.W_cache) + ((1 - self.beta_2) * layer.dW**2)
            layer.b_cache = (self.beta_2 * layer.b_cache) + ((1 - self.beta_2) * layer.db**2)
            #
            W_cache_correct = layer.W_cache / (1 - self.beta_2 ** (self.epoch + 1))
            b_cache_correct = layer.b_cache / (1 - self.beta_2 ** (self.epoch + 1))
            #
            W_updates = -self.current_learning_rate * W_momentum_correct / (np.sqrt(W_cache_correct) + self.epsilon)
            b_updates = -self.current_learning_rate * b_momentum_correct / (np.sqrt(b_cache_correct) + self.epsilon)
            #
            layer.W += W_updates
            layer.b += b_updates
        elif(type(layer).__name__ == 'RNN') :
            if not hasattr(layer, 'Waa_cache') :
                layer.Waa_momentum = np.zeros_like(layer.Waa)
                layer.Wax_momentum = np.zeros_like(layer.Wax)
                layer.ba_momentum = np.zeros_like(layer.ba)
                layer.Waa_cache = np.zeros_like(layer.Waa)
                layer.Wax_cache = np.zeros_like(layer.Wax)
                layer.ba_cache = np.zeros_like(layer.ba)
            #
            layer.Waa_momentum = (self.beta_1 * layer.Waa_momentum) + ((1 - self.beta_1) * layer.dWaa)
            layer.Wax_momentum = (self.beta_1 * layer.Wax_momentum) + ((1 - self.beta_1) * layer.dWax)
            layer.ba_momentum = (self.beta_1 * layer.ba_momentum) + ((1 - self.beta_1) * layer.dba)
            #
            Waa_momentum_correct = layer.Waa_momentum / (1 - self.beta_1 ** (self.epoch + 1))
            Wax_momentum_correct = layer.Wax_momentum / (1 - self.beta_1 ** (self.epoch + 1))
            ba_momentum_correct = layer.ba_momentum / (1 - self.beta_1 ** (self.epoch + 1))
            #
            layer.Waa_cache = (self.beta_2 * layer.Waa_cache) + ((1 - self.beta_2) * layer.dWaa**2)
            layer.Wax_cache = (self.beta_2 * layer.Wax_cache) + ((1 - self.beta_2) * layer.dWax**2)
            layer.ba_cache = (self.beta_2 * layer.ba_cache) + ((1 - self.beta_2) * layer.dba**2)
            #
            Waa_cache_correct = layer.Waa_cache / (1 - self.beta_2 ** (self.epoch + 1))
            Wax_cache_correct = layer.Wax_cache / (1 - self.beta_2 ** (self.epoch + 1))
            ba_cache_correct = layer.ba_cache / (1 - self.beta_2 ** (self.epoch + 1))
            #
            Waa_updates = -self.current_learning_rate * Waa_momentum_correct / (np.sqrt(Waa_cache_correct) + self.epsilon)
            Wax_updates = -self.current_learning_rate * Wax_momentum_correct / (np.sqrt(Wax_cache_correct) + self.epsilon)
            ba_updates = -self.current_learning_rate * ba_momentum_correct / (np.sqrt(ba_cache_correct) + self.epsilon)
            #
            layer.Waa += Waa_updates
            layer.Wax += Wax_updates
            layer.ba += ba_updates
            #
    #  
    #
    def post_update_params(self) :
        self.epoch += 1