'''
Created on 1 Apr 2017

@author: alice<aliceinnets[at]gmail.com>
'''

from abc import ABC, abstractmethod

class Model(ABC):
    
    def __init__(self, param):
        self.param = param
    
    def init(self):
        self.set()
    
    @abstractmethod
    def set(self, param):
        pass
    
    def get(self):
        return self.param
        
    