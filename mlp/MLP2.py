import numpy as np
import math
import pickle

def sigmoid(x):
    return np.tanh(x)

def dsigmoid(x):
    return 1.0-x**2

def mySigmoid(x):
    ''' Sigmoid like function using tanh '''
    return 1.0 / (1.0 + np.exp(-x))

def myDsigmoid(x):
    ''' Derivative of sigmoid above '''
    return (np.exp(-x)) / (1.0 + np.exp(-x))**2.0

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - (np.tanh(x))**2.0

def bin2dec(b):
	i=0
	n=len(b)
	puissance=0
	index=0
	while index>=-n:
		if b[index]==1:
			i= i + 2**puissance
		elif b[index]!=0 :
			return None
		puissance= puissance +1
		index=index-1
	return i

def erreurSortie(reel,attendu):
    sortie = 0
    for i in range(len(reel)):
        sortie += (reel[i]-attendu[i])**2
    return sortie


class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]

    def propagate_forward_profondeur(self, data, profondeur):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[profondeur][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        self.layers[profondeur+1][...] = sigmoid(np.dot(self.layers[profondeur],self.weights[profondeur]))

        # Return output
        return self.layers[profondeur+1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = (target - self.layers[-1])
        #error = target - self.layers[-1] + math.sqrt((self.layers[1]**2).sum())
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)
        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)


        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()

    def propagate_delta(self, target):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = (target - self.layers[-1])# + math.sqrt((self.layers[1]**2).sum()/(len(self.layers)*len(self.layers[0])))
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)
        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
        return deltas

    def update_weights(self, deltas,lrate=0.1,momentum=0.1):
         # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

    def getWeights(self):
        return self.weights

    def getLayer(self):
        return self.layers
