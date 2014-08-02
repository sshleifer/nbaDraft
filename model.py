from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import random
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations
from assemble_dataset import lh_vars, dummy_out, median_fill, drop_unnamed
import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer

clean_train = pd.read_csv('train.csv')


def rand_forest(df=clean_train, dv='tot200'):
    rfr = RandomForestRegressor()
    feature_names = get_numeric_features(df)
    target = np.array(df[dv])
    df = median_fill(df)
    
    params = {'n_estimators': [20, 30],
              'max_features': [0.1],
              'max_depth': [5, 6, 7, 8, 9, 10]}
    gs = GridSearchCV(rfr,params, cv=5, scoring='r2') 
    gs.fit(df[feature_names], target)
    
    print gs.best_params_
    print gs.best_score_
    return gs
    print cross_val_score(rfr, np.array(df[feature_names]), np.array(df.tot200), scoring='r2')
    return rfr


def get_numeric_features(colpro, col_path = 'colData.csv', meas_path = 'measurements.csv'):
    col = drop_unnamed(pd.read_csv(col_path)).columns
    meas = drop_unnamed(pd.read_csv(meas_path)).columns
    cp = drop_unnamed(colpro)
    iv_list = []

    for column in cp._get_numeric_data().columns:
        if (column in col) or (column in meas):
            iv_list.append(column)
    return iv_list


def plot_roc_curve(target_test, target_predicted_proba): 
    fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")


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

def four_way_split (df, dv=['tot200'],test_size=.2):
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


def skl_reg(X,Y):
    reg = LinearRegression()
    reg.fit(X,Y)
    return reg
