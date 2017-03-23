'''
Created on 23 Mar 2017

@author: alice<aliceinnets@gmail.com>
'''
import os
import numpy as np

class Struct:
    def __tostr__(self):
        return self.dda+'/'+self.dtype

def loadfiles_as_dict(path,ext):
    files = [f for f in os.listdir(path) if f.endswith('.'+ext)]
    
    dict = {}
    for i in range(0,len(files)):
        name = os.path.splitext(files[i])[0]
        dict[name] = np.loadtxt(path+'/'+files[i])
    
    return dict
    

def loadtxts_as_dict(path):
    return loadfiles_as_dict(path, 'txt')

def loadfiles_as_struct(path,ext):
    path = path+'/'
    files = [f for f in os.listdir(path) if f.endswith('.'+ext)]
    
    struct = Struct()
    for i in range(0,len(files)):
        name = os.path.splitext(files[i])[0]
        exec('struct.'+name+' = np.loadtxt(path+files[i])')
    
    return struct

def loadtxts_as_struct(path):
    return loadfiles_as_struct(path, 'txt')
