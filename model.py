import pandas as pd
import requests
import numpy as np
import statsmodels.api as sm
from itertools import combinations
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn import preprocessing
from assemble_dataset import make_colpro,lh_vars
import sklearn

def df_mapper(lh):
    colpro = make_colpro()
    to_map = []
    for col in lh:
        colpro[col] = colpro[col].astype('float')
        to_map.append((col,sklearn.preprocessing.StandardScaler()))
    mapper = DataFrameMapper(to_map)
    return mapper

def piping(colpro, mapper):
    pipe = sklearn.pipeline.Pipeline([('featurize', mapper), ('lm', sklearn.linear_model.LinearRegression())])
    x = cross_val_score(pipe,colpro, colpro.off100, 'r2',verbose=2)
    return np.mean(x)

def genetic_loop(colpro, num_lh, stop=1000000000000):
    ivs = lh_vars(colpro)
    rh = colpro['off100']
    #reg = regress(colpro[ivs], rh)
    #print reg.rsquared_adj
    #best_result = reg
    for i, combo in enumerate(combinations(ivs,num_lh)):
        lh = list(combo)
        mapper = df_mapper(lh)
        result = piping(colpro, mapper)
        if i == 0:
            best_result = result
        elif result.rsquared_adj > best_result.rsquared_adj:
            best_result = result
        if i == stop:
            break
    return result #Can return best_vars if useful


def regress(X, Y): # FASTER THAN PIPING, CHANGE TO RESULT>rsquared_adj
    reg = sm.OLS(Y,X)
    return reg.fit()

def full_reg(colpro):
    reg = regress(colpro[lh_vars()],colpro['off100'])
    #genetic_loop(colpro[lh_vars()],colpro['off100'])
    return reg


def best_num_vars(colpro):
    results = {}
    print len (colpro.columns)
    for i in range(1, len(colpro.columns)):
        result = genetic_loop(colpro, i, 5000)
        results.update({len(result.pvalues): result.rsquared_adj})
        print i, results[i]
    return results
