'''Attempts to predict rapm using college stats-Sam Shleifer-2 August, 2014'''

from assemble_dataset import lh_vars, dummy_out, median_fill, drop_unnamed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
CLEAN_TRAIN = pd.read_csv('train.csv')


def rand_forest(df=CLEAN_TRAIN, dv='tot200'):
    rfr = RandomForestRegressor()
    feature_names = get_numeric_features(df)
    target = np.array(df[dv])
    df = median_fill(df)

    params = {'n_estimators': [20, 30],
              'max_features': [0.1],
              'max_depth': [5, 6, 7, 8, 9, 10]}
    gs = GridSearchCV(rfr, params, cv=5, scoring='r2')
    gs.fit(df[feature_names], target)

    print gs.best_params_
    print gs.best_score_
    return gs
    print cross_val_score(rfr,
                          np.array(df[feature_names]),
                          np.array(df.tot200),
                          scoring='r2')
    return rfr

COL_PATH = 'colData.csv'
MEAS_PATH = 'measurements.csv'
def get_numeric_features(colpro):
    col = drop_unnamed(pd.read_csv(COL_PATH)).columns
    meas = drop_unnamed(pd.read_csv(MEAS_PATH)).columns
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
        to_map.append((col, StandardScaler()))
    mapper = DataFrameMapper(to_map)
    return mapper

def shuffle(df):
    return df.loc[np.random.choice(df.index, len(df), replace=False)]

def four_way_split(df, dv=['tot200'], test_size=.2):
    xtrain, xtest, ytrain, ytest = train_test_split(df[lh_vars(df)],
                                                    df[dv],
                                                    test_size=test_size,
                                                    random_state=0)
    xtrain = pd.DataFrame(xtrain, columns=lh_vars(df))
    xtest = pd.DataFrame(xtest, columns=lh_vars(df))
    ytrain = pd.DataFrame(ytrain, columns=[dv])
    ytest = pd.DataFrame(ytest, columns=[dv])
    return xtrain, xtest, ytrain, ytest


def rfr_regression(full=CLEAN_TRAIN, dv='tot200', test_size=.7,
                   add_rfr=True, add_off=True, add_def=True):
    df = dummy_out(full.copy())
    xtr, xt, ytr, yt = four_way_split(df, test_size=test_size)
    if add_rfr:
        rfr = RandomForestRegressor()
        rfr.fit(np.array(xtr), np.array(ytr[dv].values))
        xt['forest_pred'] = rfr.predict(np.array(xt.values))
        xtr['forest_pred'] = rfr.predict(np.array(xtr.values))
    if add_off:
        off_reg = skl_reg(xtr, ytr['off100'])
        xt['off_pred'] = off_reg.predict(np.array(xt.values))
        xtr['off_pred'] = off_reg.predict(xtr)
    if add_def:
        def_reg = skl_reg(xtr, ytr['def100'])
        xt['def_pred'] = def_reg.predict(np.array(xt.values))
        xtr['def_pred'] = def_reg.predict(xtr)
    reg = skl_reg(xtr, ytr[dv])
    return reg.score(np.array(xt), np.array(yt[dv]))


def skl_reg(X, Y):
    reg = LinearRegression()
    reg.fit(X, Y)
    return reg
