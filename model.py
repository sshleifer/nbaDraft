from scipy.stats import pearsonr
import pandas as pd
import requests
import numpy as np
import statsmodels.api as sm
from itertools import combinations
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn import preprocessing
from assemble_dataset import make_colpro,lh_vars, dummy_out
import sklearn

def df_mapper(lh):
    colpro = make_colpro()
    to_map = []
    for col in lh:
        colpro[col] = colpro[col].astype('float')
        to_map.append((col,sklearn.preprocessing.StandardScaler()))
    mapper = DataFrameMapper(to_map)
    return mapper

def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

def training_regression(colpro,num_folds=2):
    colpro = shuffle(colpro)
    obs = len(colpro)
    x_train =  colpro[:obs/num_folds]
    x_test = colpro[obs/num_folds:]
    reg,lh = prune(x_test)
    y_hat = list(reg.predict(x_test[lh]))
    #print y_hat
    y_act = pd.DataFrame(x_test['tot200'])
    y_act['y_hat'] = [n for n in y_hat]
    print pearsonr(y_act.y_hat, y_act.tot200)
    return reg, y_act

def genetic_loop(colpro, num_lh, dv='off100', stop=1000000000000):
    ivs = lh_vars(colpro)
    rh = colpro[dv]
    for i, combo in enumerate(combinations(ivs,num_lh)):
        lh = list(combo)
        result = regress(colpro[lh], rh)
        if i == 0:
            best_result = result
        elif result.rsquared_adj > best_result.rsquared_adj:
            best_result = result
        if i == stop:
            break
    return result #Can return best_vars if useful

def prune(colpro, cutoff=.3,dv='tot200',iters=5):
    reg = full_reg(dummy_out(colpro), dv)
    while iters > 0:
        p = pd.DataFrame(reg.pvalues,columns=['pvals'])
        lh_new =  list(p[p.pvals <= cutoff].index)
        reg = regress(colpro[lh_new],colpro[dv])
        iters -= 1
    return reg, lh_new

def regress(X, Y): # FASTER THAN PIPING, CHANGE TO RESULT>rsquared_adj
    reg = sm.OLS(Y,X)
    return reg.fit()

def full_reg(colpro, dv='off100'):
    reg = regress(colpro[lh_vars(colpro)],colpro[dv])
    return reg

def best_num_vars(colpro):
    results = {}
    print len (colpro.columns)
    for i in range(1, len(colpro.columns)):
        result = genetic_loop(colpro, i, 5000)
        results.update({len(result.pvalues): result.rsquared_adj})
        print i, results[i]
    return results
