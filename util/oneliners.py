'''
Created on 23 Mar 2017

@author: alice<aliceinnets[at]gmail.com>
'''
import os
import numpy as np

HOME_PATH = os.path.expanduser('~')
RESULTS_PATH = HOME_PATH+'/results/'
TEST_RESULTS_PATH = HOME_PATH+'/temp/'
DATA_PATH = HOME_PATH+'/data/'

class struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    
    def keys(self):
        return [name for name in vars(self) if not name.startswith('_')]
        

def loadtxts_to_dict(path):
    path += '/'
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    
    data = {}
    for i in range(0,len(files)):
        name = os.path.splitext(files[i])[0]
        data[name] = np.loadtxt(path+'/'+files[i])
    
    return data

def loadtxts_to_struct(path):
    path += '/'
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    
    data = struct()
    for i in range(0,len(files)):
        name = os.path.splitext(files[i])[0]
        exec('data.'+name+' = np.loadtxt(path+files[i])')
    
    return data

def savetxts_from_dict(path, data):
    path += '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    for name in data.keys():
        np.savetxt(path+str(name)+'.txt', data[name])

def savetxts_from_struct(path, data):
    path += '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    for name in data.keys():
        exec('np.savetxt(path+\''+name+'.txt\', data.'+name+')')

def remove_all(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            if os.path.isfile(path+file):
                os.remove(path+file)
            else:
                remove_all(path+file+'/')
        os.rmdir(path)
        
    
    