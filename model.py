'''Attempts to predict rapm using college stats-Sam Shleifer-August 2nd, 2014'''

from assemble_dataset import lh_vars, dummy_out, median_fill, drop_unnamed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

CLEAN_TRAIN = pd.read_csv('train.csv')

def pipe_attempt(df=CLEAN_TRAIN, dv='tot200'):
    df = df.fillna(-1)
    feature_names = get_numeric_features(df)
    print len(feature_names)
    target = df[dv]
    imputer = Imputer(strategy='median', missing_values=-1)
    regressor = RandomForestRegressor(n_estimators=100, max_features=.25)
    xtr, xt, ytr, yt = train_test_split(df[feature_names].values, df[dv].values)
    imputer.fit(xtr)
    xtr_imputed = imputer.transform(xtr)
    r = regressor.fit(xtr_imputed, np.ravel(ytr))
    #return df[feature_names], xtr_imputed
    feature_plot(r, feature_names[1:])
    #regressor = LinearRegression()
    pipeline = Pipeline([
        ('imp', imputer),
        ('clf', regressor),
    ])
    scores = cross_val_score(pipeline, df[feature_names].values, target.values, cv=5)
    print scores.mean()
    rfr_params = {
            'imp__strategy' : ['mean', 'median', 'most_frequent'],
            'clf__max_features' : [0.1, 0.25, 1],
            'clf__max_depth' : [10, 20, 30]}
    lin_params = {
            'imp__strategy' : ['mean', 'median', 'most_frequent'],
            'clf__normalize': ['True', 'False']}
    xtr, xt, ytr, yt = four_way_split(df[feature_names], target)
    #gs = GridSearchCV(pipeline, lin_params, cv=5)
    gs = GridSearchCV(pipeline, rfr_params, cv=5)
    gs.fit(xtr, np.ravel(ytr))
    return gs

#Best = { max_depth: 10, max_features: .25, imp: mean}

def gridded_rfr():
    rfr = RandomForestRegressor(n_estimators=100, max_features=.25)
    return rfr

def feature_plot(clf, feature_names):
    x = np.arange(len(feature_names))
    plt.bar(x, clf.feature_importances_)
    _ = plt.xticks(x + 0.5, feature_names)
    plt.show()

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

def four_way_split(x, y, test_size=.2):
    xtrain, xtest, ytrain, ytest = train_test_split(x.values, y.values, test_size=test_size)
    xtrain = pd.DataFrame(xtrain, columns=x.columns)
    xtest = pd.DataFrame(xtest, columns=x.columns)
    ytrain = pd.DataFrame(ytrain, columns=['dv'])
    ytest = pd.DataFrame(ytest, columns=['dv'])
    return xtrain, xtest, ytrain, ytest

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



def rfr_regression(full=CLEAN_TRAIN, dv='tot200', test_size=.7,
                   add_rfr=True, add_off=True, add_def=True):
    df = dummy_out(full.copy())
    xtr, xt, ytr, yt = four_way_split(df[get_numeric_features(df)].values, df[dv].values, test_size=test_size)
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
