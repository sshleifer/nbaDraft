import pandas as pd
import requests
import numpy as np
#import statsmodels.api as sm
#from itertools import combinations
from sklearn_pandas import DataFrameMapper, cross_val_score
from scrapeDX import make_colpro
from sklearn import preprocessing
from scrapeDX import make_colpro,lh_vars
import sklearn

def df_mapper(lh):
    colpro = make_colpro()
    to_map = []
    for col in lh:
        colpro[col] = colpro[col].astype('float')
        to_map.append((col,sklearn.preprocessing.StandardScaler()))
    mapper = DataFrameMapper(to_map)
    return mapper.fit_transform(colpro)

def piping(colpro, mapper):
    pipe = sklearn.pipeline.Pipeline([('featurize', mapper), ('lm', sklearn.linear_model.LinearRegression())])
    x = cross_val_score(pipe,colpro, colpro.off100, 'r2',verbose=2)
    return x


def genetic_loop(colpro, num_lh, stop=1000000000000):
    ivs = lh_vars(colpro)
    rh = colpro['off100']
    #reg = regress(colpro[ivs], rh)
    #print reg.rsquared_adj
    #best_result = reg
    for i, combo in enumerate(combinations(ivs,num_lh)):
        lh = list(combo)
        result = regress(colpro[lh], rh)
        if i == 0:
            best_result = result
            continue
        if result.rsquared_adj > best_result.rsquared_adj:
            best_result = result
        if i == stop:
            break
    return result #Can return best_vars if useful
