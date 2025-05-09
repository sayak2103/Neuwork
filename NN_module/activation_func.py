#classes that define the activation functions
import numpy as np
class Activation : 
    #
    def get_activation(self, act='linear') : 
        if(act=='linear') :
            return Activation().Linear()
        elif(act=='sigmoid') :
            return Activation().Sigmoid()
        elif(act=='relu') :
            return Activation().Relu()
        elif(act=='softmax') :
            return Activation().Softmax()
        elif(act=='tanh') :
            return Activation().Tanh()
        else :
            raise ValueError("Activation function not found")
    #
    #creating activations as seperate nested classes
    class Sigmoid : 
        # input for the this function is z = a @ w + b
        def activate(self, z) :
            return 1 / ( 1 + np.exp(-z) )

        #input for this function is activated matrix g(z)
        def get_g_grad(self, g_z , grad) : 
            return ((g_z * ( 1 - g_z )) * grad)
    #
    class Linear : 
        def activate(self, z) : 
            return z

        def get_g_grad(self, g_z , grad) :
            return grad
    #
    class Relu :
        def activate(self, z) :
            relu = np.maximum(0, z)
            return relu

        def get_g_grad(self, g_z , grad) :
            relu_diff = np.ones_like(g_z)
            relu_diff[g_z <= 0] = 0
            #relu_diff = lambda x : 1 if x>0 else 0
            #relu_diff = np.vectorize(relu_diff)
            return relu_diff * grad
    #
    class Softmax :
        def activate(self, z) :
            g_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            s = np.sum(g_z, axis=1, keepdims=True)
            g_z = g_z/s
            return g_z
        
        #for the last layer to be softmax the cost function must be categoricalCrossEntropy
        def get_g_grad(self, g_z , grad) :
            #this is used when a combined derivative of softmax and cce is used in Cce class
            #return np.ones_like(g_z)

            #seperate derivative of softmax
            dinputs = np.empty_like(g_z)
            # Enumerate outputs and gradients
            for index, (single_output, single_dvalues) in enumerate ( zip (g_z, grad)):
                # Flatten output array
                single_output = single_output.reshape( - 1 , 1 )
                # Calculate Jacobian matrix of the output
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                # Calculate sample-wise gradient
                # and add it to the array of sample gradients
                dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
            #
            return dinputs
    #
    class Tanh :
        def activate(self, z) :
            return np.tanh(z)

        def get_g_grad(self, g_z , grad) :
            return (1 - np.square(g_z)) * grad
    #