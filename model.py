from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import random
from scipy.stats import pearsonr
import pandas as pd
import requests
import numpy as np
import statsmodels.api as sm
from itertools import combinations
from assemble_dataset import set_up,lh_vars, dummy_out
import sklearn

#df = set_up()
clean_train = pd.read_csv('train.csv')

def rand_forest(df=clean_train):
    df = dummy_out(df)
    rfr = RandomForestRegressor()
    print cross_val_score
    xtr, xt, ytr, yt = four_way_split(df)
    rfr.fit(np.array(xtr.values), np.array(ytr.values))
    return rfr, score_forest(rfr, xt, yt)

def score_forest(rfr,xt,yt):
    pred = rfr.predict(xt)
    y= make_scoreable(yt,pred)
    return fit_score(y)

def normalize(x):
    m = x.mean()
    s = x.std()
    return [(a-m)/s for a in x.values]

def norm_df(df):
    for col in df._get_numeric_data().columns:
        df[col] = normalize(df[col])
    return df

def df_mapper(lh, df):
    to_map = []
    for col in lh:
        df[col] = df[col].astype('float')
        to_map.append((col,sklearn.preprocessing.StandardScaler()))
    mapper = DataFrameMapper(to_map)
    return mapper

def shuffle(df):
    return df.loc[np.random.choice(df.index, len(df), replace=False)] 

def four_way_split (df, dv=['off100','def100','tot200'],test_size=.5):
    df = shuffle(df) 
    xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(
            df[lh_vars(df)],
            df[dv],
            test_size = test_size,
            random_state = 0)
    xtrain = pd.DataFrame(xtrain, columns = lh_vars(df))
    xtest = pd.DataFrame(xtest, columns = lh_vars(df))
    ytrain = pd.DataFrame(ytrain, columns = [dv])
    ytest = pd.DataFrame(ytest, columns = [dv])
    return xtrain, xtest, ytrain, ytest

def two_way_split(df, test_size=.5):
    df = df.loc[np.random.choice(df.index, len(df), replace=False)] 
    train_size = np.round(len(df) * (1 - test_size))
    rows = random.sample(df.index, int(train_size))
    train = df.ix[rows]
    test = df.drop(rows)
    return train, test

def rfr_regression(full=clean_train, dv='tot200', test_size=.7, 
                   add_rfr=True, add_off=True, add_def=True):
    df = dummy_out(full.copy())
    xtr, xt, ytr, yt = four_way_split(df, test_size=test_size)
    if add_rfr: 
        rfr = RandomForestRegressor()
        rfr.fit(np.array(xtr),np.array(ytr[dv].values))
        xt['forest_pred'] =rfr.predict(np.array(xt.values)) 
        xtr['forest_pred'] = rfr.predict(np.array(xtr.values))
    if add_off:
        off_reg= skl_reg(xtr,ytr['off100'])
        xt['off_pred'] =off_reg.predict(np.array(xt.values)) 
        xtr['off_pred'] = off_reg.predict(xtr)
    if add_def:
        def_reg = skl_reg(xtr,ytr['def100'])
        xt['def_pred'] = def_reg.predict(np.array(xt.values)) 
        xtr['def_pred'] = def_reg.predict(xtr)
#    reg = skl_reg(xtr, ytr['tot200'])
    reg = skl_reg(xtr, ytr[dv])
    return reg.score(np.array(xtr), np.array(ytr[dv]))
    print np.mean(cross_val_score(reg, xt, yt, scoring='r2'))
    return regress(xtr, ytr[dv])

def run_prune(df,dv='tot200',test_size=.5):
    df = dummy_out(df)
    test, train = two_way_split(df, test_size=test_size) 
    reg, lh = prune(test)
    yhat = list(reg.predict(test[lh]))
    y = pd.DataFrame(test.tot200)
    y = make_scoreable(y,yhat)
    print fit_score(y)
    return reg, lh

def make_scoreable(df, pred):
    df['yhat'] = [x for x in pred]
    return df

def fit_score(y):
    return pearsonr(y.tot200, y.yhat)[0] 

def genetic_loop(df, num_lh, dv='off100', stop=1000000000000):
    ivs = lh_vars(df)
    rh = df[dv]
    for i, combo in enumerate(combinations(ivs,num_lh)):
        lh = list(combo)
        result = regress(df[lh], rh)
        if i == 0:
            best_result = result
        elif result.rsquared_adj > best_result.rsquared_adj:
            best_result = result
        if i == stop:
            break
    return result #Can return best_vars if useful

def prune(df=clean_train, cutoff=.3,dv='tot200',iters=5):
    reg = full_reg(df, dv)
    while iters > 0:
        p = pd.DataFrame(reg.pvalues,columns=['pvals'])
        lh_new =  list(p[p.pvals <= cutoff].index)
        reg = regress(df[lh_new],df[dv])
        iters -= 1
        print 'itered'
    return reg, lh_new

def regress(X, Y): # FASTER THAN PIPING, CHANGE TO RESULT>rsquared_adj
    reg = sm.OLS(Y,X)
    return reg.fit()

def skl_reg(X,Y):
    reg = LinearRegression()
    reg.fit(X,Y)
    return reg

def full_reg(df, dv='off100'):
    reg = regress(df[lh_vars(df)],df[dv])
    return reg
