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

from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split

from time import gmtime, strftime

from util import oneliners

DATA_FOLDER = oneliners.dirname(__file__)+"/House Prices Advanced Regression Techniques/"

def main(_):
    data_train = pd.read_csv(DATA_FOLDER+"train.csv")
    data_test = pd.read_csv(DATA_FOLDER+"test.csv")
    
    dim = 7
#     columns = component_analysis_by_corr(data_train, dim)
#     columns = component_analysis_by_linear_model();
    columns = component_analysis_by_selectKBest()
    
    ## Selected columns: ['OverallQual', 'BsmtFullBath', 'FullBath', 'KitchenAbvGr', 'GarageCars'] by linear model
    ## Selected columns: ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF']
    ## columns = ['OverallQual', 'BsmtFullBath', 'FullBath', 'KitchenAbvGr', 'GarageCars']
    
    data_train = np.array(data_train[columns])
    data_test = np.array(data_test[columns[1:]])
    
    y = data_train[:,0]
    X = data_train[:,1:]
    X_test = data_test
    
    X, X_test = fill_nan_as_mean(X, X_test)
    
    X_train, X_train_test, y_train, y_train_test = train_test_split(X, y, random_state=42, test_size=.33)
    
    y_train_mean, y_train_cov, gp_train = inference_by_gp(X_train, y_train, X_train_test)
    y_mean, y_cov, gp = inference_by_gp(X_train, y_train, X_test)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(y_train_test,'.')
    plt.plot(y_train_mean)
    plt.grid()
    plt.subplot(2,2,2)
    plt.scatter(y_train_test,y_train_mean)
    plt.grid()
    plt.subplot(2,2,3)
    plt.plot(y_mean)
    plt.grid()
    plt.show()
    
#     filename, pyfilename = save(y_mean)    

def component_analysis_by_corr(data_train, dim):    
    corr_train = data_train.corr()
    component_train = corr_train["SalePrice"].sort_values(ascending=False)
    
    return component_train[1:1+dim].index

def component_analysis_by_linear_model():
    train = pd.read_csv(DATA_FOLDER+'train.csv')
    data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    X = data.drop(['SalePrice', 'Id'], axis=1)
    y = np.log(train.SalePrice)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

    lr = linear_model.LinearRegression()

    rfe = RFE(lr, 5)
    fit = rfe.fit(X_train, y_train)
    print("Features: {features}".format(features=X.columns))
    print("Num Features: {number_features}".format(number_features=fit.n_features_))
    print("Selected Features: {support}".format(support=fit.support_))
    print("Feature Ranking: {ranking}".format(ranking=fit.ranking_))

    selected_columns = [column for column, selected in zip(X.columns, fit.support_) if selected]
    print("Selected columns: {selected}".format(selected = selected_columns))
    
    return selected_columns

def component_analysis_by_selectKBest():
    train = pd.read_csv(DATA_FOLDER+'train.csv')
    data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    X = data.drop(['SalePrice', 'Id'], axis=1)
    y = np.log(train.SalePrice)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

    test = SelectKBest(score_func=f_regression, k=4)
    fit = test.fit(X, y) # ValueError: Unknown label type - https://stackoverflow.com/questions/34246336/python-randomforest-unknown-label-error - only when using chi2 function
    # fit = test.fit(X, np.asarray(y, dtype="|S6"))

    np.set_printoptions(precision=3, suppress=True)

    print("Features: {features}".format(features=X.columns))
    print("Scores: {scores}".format(scores=fit.scores_))

    values = [(value, float(score)) for value, score in sorted(zip(X.columns, fit.scores_), key=lambda x: x[1] * -1)]
    #print(tabulate(values, ["column", "score"], tablefmt="plain", floatfmt=".4f"))

    selected_features = fit.transform(X)
    print("Features: {selected_features}".format(selected_features = selected_features))

    return selected_features
    
def fill_nan_as_mean(X, X_test): 
    X_mean_test = np.nanmean(X_test,axis=0)
    nan_index = np.where(np.isnan(X_test))
    for i in range(len(nan_index[0])):
        X_test[nan_index[0][i],nan_index[1][i]] = X_mean_test[nan_index[1][i]]
    
    return X, X_test
    
def inference_by_gp(X, y, X_test):
    sigma_f = np.mean(y)
    # sigma_f = np.std(y)
    sigma_f_low = sigma_f*1e-2
    sigma_f_high = sigma_f*1e2
    
    sigma_x = np.mean(X)
    sigma_x_low = sigma_x*1e-2
    sigma_x_high = sigma_x*1e2
    sigma_x = np.mean(X, axis = 0)
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
    
    return y_mean, y_cov, gp

def plot(filenames):
    plt.figure()
    for i in range(len(filenames)):
        submission = (pd.read_csv(DATA_FOLDER+filenames[i]))
        plt.plot(submission["SalePrice"])
    

def save(y_test):
    time = strftime("%Y%m%d%H%M%S", gmtime())
    filename = "submission-"+time+".csv"
    pyfilename = "submission-"+time+".py"
    
    submission = pd.read_csv(DATA_FOLDER+"sample_submission.csv")
    submission["SalePrice"] = y_test
    submission.to_csv(DATA_FOLDER+filename, index=False)
    
    shutil.copy(os.path.abspath(__file__), DATA_FOLDER+pyfilename)
    
    return filename, pyfilename
    
    
if __name__ == "__main__": main(_)
