# losses file has the loss functiuon
import numpy as np
class Losses :
    def get_loss(self, loss) :
        if(loss == 'mse') :
            return Losses().Mse()
        elif(loss == 'categoricalCrossentropy') :
            return Losses().Cce()
        elif(loss == 'binaryCrossentropy') :
            return Losses().Bce()
    
    #each  loss function is desigened as seperate class
    class Mse :
        def compute_cost(self, y_cap, y) :
            cost = np.mean((y_cap-y)**2)
            return cost

        def get_grad(self, y_cap, y) :
            m = len(y_cap)
            outputs = len(y_cap[0])
            grad = 2 * (y_cap-y)
            grad /= m
            grad /= outputs
            return grad

    #loss function Categorical Cross Entropy
    class Cce :
        #categorical cross entropy must be used for multiclass classification 
        # therefore, y can store the number of class that a sample belongs to, (sparse) OR 
        # y can also be one hot coded vector
        #n_k also being the number of neurons on the last softmax layer
        #ategorical Cross Entroy be used with softmax on the output layer of any neuwork
        def compute_cost(self, y_cap, y) :
            y_cap_clipped = np.clip(y_cap,1e-7,1-1e-7)
            
            if(len(y.shape)==1) :
                categorical_prob = y_cap[range(len(y_cap_clipped)), y]
            else :
                categorical_prob = np.sum(y_cap_clipped * y, axis=1)
            
            neg_log = -np.log(categorical_prob)
            cost = np.mean(neg_log)
            return cost

        def get_grad(self, y_cap, y) :
            m = len(y_cap) #number of samples
            if(len(y.shape)==2) :
                y = np.argmax(y, axis=1)
            grad = y_cap.copy()
            grad[range(m) , y] -= 1
            grad /= m
            return grad
            
            ###
            #classes = len(y_cap[0])
            ##conversion of y into a aonehot vector if needed
            #if(len(y.shape)==1) :
            #    y = np.eye(classes)[y]
            #grad = -y/y_cap
            #grad /= m #normalization
            #return grad
            ###

    class Bce :
        #generally logarithm of maximum likelihood is used for binary classification along with sigmoi in the outout layer
        def compute_cost(self, y_cap, y) :
            y_cap_clipped = np.clip(y_cap,1e-7,1 - 1e-7)
            cost = -(y * np.log(y_cap_clipped) + (1-y) * (1 - np.log(1 - y_cap_clipped)))
            cost = np.mean(cost, axis=-1)
            return cost
        
        def get_grad(self, y_cap, y) :
            m = len(y_cap)
            labels = len(y_cap[0])
            y_cap_clipped = np.clip(y_cap, 1e-7, 1 - 1e-7)
            grad = -(y / y_cap_clipped - (1 - y)/(1 - y_cap_clipped))
            grad /= labels
            grad /= m
            return grad