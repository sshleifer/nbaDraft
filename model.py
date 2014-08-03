'''Attempts to predict rapm using college stats-Sam Shleifer-August 2nd, 2014'''

from assemble_dataset import drop_unnamed
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
COL_PATH = 'datasets/colData.csv'
MEAS_PATH = 'datasets/measurements.csv'

def rfr(train=CLEAN_TRAIN, target_var='tot200'):
    '''Best model so far: RandomForestRegressor'''
    train = train.fillna(-1)
    feature_names = get_numeric_features(train)
    X = train[feature_names].values
    target = train[target_var].values
    x_train, x_test, y_train, y_test = train_test_split(X, target)
    imputer = median_imputer(x_train)
    x_train_imputed = imputer.transform(x_train)
    regressor = RandomForestRegressor(n_estimators=100, max_features=.25)
    regressor.fit(x_train_imputed, np.ravel(y_train))
    feature_plot(regressor, feature_names[1:])
    print np.mean(cross_val_score(regressor, X, target, cv=5))
    return regressor


def feature_plot(clf, feature_names): #TODO FEATURE_NAME CROWDING
    '''Plots feature importances associated with clf'''
    x_axis = np.arange(len(feature_names))
    plt.bar(x_axis, clf.feature_importances_)
    _ = plt.xticks(x_axis + 0.5, feature_names)
    plt.show()


def median_imputer(x_train):
    '''fits the median imputer'''
    imputer = Imputer(strategy='median', missing_values=-1)
    imputer.fit(x_train)
    return imputer


def make_pipeline(train=CLEAN_TRAIN, target_var='tot200'):
    '''Makes the median imputer -> rfr pipeline'''
    train = train.fillna(-1)
    feature_names = get_numeric_features(train)
    X = train[feature_names].values
    target = train[target_var].values
    x_train, x_test, y_train, y_test = train_test_split(X, target)
    imputer = median_imputer(x_train)
    regressor = RandomForestRegressor(n_estimators=100, max_features=.25)

    ###GRID SEARCH###
    pipeline = Pipeline([
        ('imp', imputer),
        ('clf', regressor),
    ])
    return X, target, pipeline


def grid_search(info_tuple=None):
    '''Tests various RFR and imputer params'''
    if not info_tuple:
        info_tuple = make_pipeline()
    X, target, pipeline = info_tuple
    x_train, x_test, y_train, y_test = train_test_split(X, target)
    scores = cross_val_score(pipeline, X, target, cv=5)
    print scores.mean()
    rfr_params = {'imp__strategy' : ['mean', 'median', 'most_frequent'],
                  'clf__max_features' : [0.1, 0.25, 1],
                  'clf__max_depth' : [10, 20, 30]}
    lin_params = {'imp__strategy' : ['mean', 'median', 'most_frequent'],
                  'clf__normalize': ['True', 'False']}
    #gs = GridSearchCV(pipeline, lin_params, cv=5)
    gs = GridSearchCV(pipeline, rfr_params, cv=5)
    gs.fit(x_train, np.ravel(y_train))
    return gs  #Best = { max_depth: 10, max_features: .25, imp: mean}


def get_numeric_features(colpro):
    '''Gets features that were not created after player was drafted'''
    col = drop_unnamed(pd.read_csv(COL_PATH)).columns
    meas = drop_unnamed(pd.read_csv(MEAS_PATH)).columns
    cp = drop_unnamed(colpro)
    iv_list = []

    for column in cp._get_numeric_data().columns:
        if (column in col) or (column in meas):
            iv_list.append(column)
    return iv_list


def plot_roc_curve(target_test, target_predicted_proba):
    '''For classification'''
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
    '''Mimics StandardScaler'''
    mean = x.mean()
    stdev = x.std()
    return [(val - mean) / stdev for val  in x.values]


def norm_df(features):
    '''normalizes all feature columns'''
    for col in features:
        features[col] = normalize(features[col])
    return features


def df_mapper(df, features):
    '''some sklearn_pandas thingy I don't use'''
    to_map = []
    for col in features:
        df[col] = df[col].astype('float')
        to_map.append((col, StandardScaler()))
    mapper = DataFrameMapper(to_map)
    return mapper


def shuffle(data):
    '''rearranges data, for cross validating by hand'''
    return data.loc[np.random.choice(data.index, len(data), replace=False)]
