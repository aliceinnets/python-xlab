'''
Created on 14 Aug 2017

@author: alice<aliceinnets[at]gmail.com>
'''

import numpy as np
import matplotlib.pyplot as plt
import os, shutil

import pandas as pd

from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from time import gmtime, strftime

from util import oneliners

DATA_FOLDER = oneliners.dirname(__file__)+"/House Prices Advanced Regression Techniques/"

df_train = pd.read_csv(DATA_FOLDER+"train.csv")
df_test = pd.read_csv(DATA_FOLDER+"test.csv")
 
corr_train = df_train.corr()
component_train = corr_train["SalePrice"].sort_values(ascending=False)[:1+5]
 
data_train = np.array(df_train[component_train.index])
data_test = np.array(df_test[component_train[1:1+5].index])
 
y = data_train[:,0]
X = data_train[:,1:]
X_test = data_test
 
X_mean_test = np.nanmean(X_test,axis=0)
nan_index = np.where(np.isnan(X_test))
for i in range(len(nan_index[0])):
    X_test[nan_index[0][i],nan_index[1][i]] = X_mean_test[nan_index[1][i]]
 
sigma_f = np.mean(y)
# sigma_f = np.std(y)
sigma_f_low = sigma_f*1e-4
sigma_f_high = sigma_f*1e1

sigma_x = np.mean(X)
sigma_x_low = sigma_x*1e-4
sigma_x_high = sigma_x*1e1
# sigma_x = np.std(X, axis = 0)
# sigma_x_low = np.mean(sigma_x*1e-1)
# sigma_x_high = np.mean(sigma_x*1e1)
 
sigma_y = sigma_f*1e-3
sigma_y_low = sigma_f_low*1e-3
sigma_y_high = sigma_f_high*1e-3
 
# kernel = sigma_f * RBF(length_scale=sigma_x) + WhiteKernel(noise_level=sigma_y)
kernel = sigma_f * RBF(length_scale=sigma_x, length_scale_bounds=(sigma_x_low, sigma_x_high)) + WhiteKernel(noise_level=sigma_y, noise_level_bounds=(sigma_y_low, sigma_y_high))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=False).fit(X, y)
 
y_mean, y_cov = gp.predict(X_test, return_cov=True)
 
# sigma_f_num_grid = 25
# sigma_x_num_grid = 5
#   
# sigma_f_grid = np.linspace(sigma_f_low, sigma_f_high, sigma_f_num_grid)
# sigma_x_grid = np.linspace(sigma_x_low, sigma_x_high, sigma_x_num_grid)
# sigma_f_mesh, sigma_x_mesh = np.meshgrid(sigma_f_grid, sigma_x_grid) 
#   
# log_evidence = [[gp.log_marginal_likelihood(np.log([sigma_f_mesh[i,j],sigma_x_mesh[i,j],1e-3*sigma_f_mesh[i,j]])) for i in range(sigma_f_mesh.shape[0])] for j in range(sigma_f_mesh.shape[1])]
# log_evidence = np.array(log_evidence).T


time = strftime("%Y%m%d%H%M%S", gmtime())
submission = pd.read_csv(DATA_FOLDER+"sample_submission.csv")
submission["SalePrice"] = y_mean
submission.to_csv(DATA_FOLDER+"submission-"+time+".csv", index=False)
shutil.copy(os.path.abspath(__file__), DATA_FOLDER+"submission-"+time+".py")
