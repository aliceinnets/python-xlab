'''
Created on 1 Apr 2017

@author: alice<aliceinnets[at]gmail.com>
'''

import numpy as np
from abc import ABC, abstractmethod

class model(ABC):
    
    def __init__(self, parameters={}):
        self.parameters = parameters
    
    def set_parameters(self, parameters):
        self.parameters = parameters
        
    def set_parameter(self, name, parameter):
        self.parameters[name] = parameter
    
    def get_parameters(self):
        return self.parameters
    
    def get_parameter(self, name):
        return self.parameter[name]
    
    @abstractmethod
    def init(self, *args):
        pass
        
    @abstractmethod
    def pred(self, x):
        pass


class perceptron(model):
    
    def init(self, *args):
        self.parameters['W'] = np.zeros(args[0])
        self.parameters['b'] = np.zeros(args[0][1])
    
    def pred(self, x):
        return np.matmul(x,self.parameters['W']) + self.parameters['b']


p = perceptron()
p.init([10, 20])
x = np.zeros([10])
print(p.get_parameters())
print(x)
print(p.pred(x))