'''
Created on 23 Mar 2017

@author: alice<aliceinnets@gmail.com>
'''

import util.oneliners as oneliners
import numpy as np

path = oneliners.test_results_path+'/game2-2/'
data1 = oneliners.loadtxts_to_dict(path)
data2 = oneliners.loadtxts_to_struct(path)

# print(data1['a'])
print(data1.keys())

# print(data2.a) 
print(data2.keys())

path = oneliners.test_results_path+'/adict/game2-2/'
oneliners.remove_all(path)
oneliners.savetxts_from_dict(path, data1)

data11 = oneliners.loadtxts_to_dict(path)

path = oneliners.test_results_path+'/astruct/game2-2/'
oneliners.remove_all(path)
oneliners.savetxts_from_struct(path, data2)

data21 = oneliners.loadtxts_to_struct(path)

print(data11.keys())
print(data21.keys())

path = oneliners.test_results_path+'/adict/'
oneliners.remove_all(path)

path = oneliners.test_results_path+'/astruct/'
oneliners.remove_all(path)

