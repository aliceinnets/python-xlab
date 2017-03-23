'''
Created on 23 Mar 2017

@author: alice<aliceinnets@gmail.com>
'''
from os.path import expanduser
from util.oneliners import loadtxts_as_dict, loadtxts_as_struct

path = expanduser("~")+'/temp/game2-2/'
dict = loadtxts_as_dict(path)
struct = loadtxts_as_struct(path)

print(dict['a'])
print(struct.a) 
