'''Assembles and preprocesses the colPro.csv, the dataset used for analysis'''
import numpy as np
import pandas as pd
MEAS_PATH = 'datasets/measurements.csv'
COL_PATH = 'datasets/colData.csv'
BEST_RAPM_PATH = 'datasets/bestRapm.csv'
COL_MEAS_PATH = 'datasets/colmeas.csv'

def numeric_feature_names(colpro):
    '''model.py's get_feature_names is a better version'''
    col = drop_unnamed(pd.read_csv(COL_PATH))._get_numeric_data().columns
    meas = drop_unnamed(pd.read_csv(MEAS_PATH))._get_numeric_data().columns
    candidates = drop_unnamed(colpro)._get_numeric_data().columns
    iv_list = []
    for column in candidates:
        if column in col or column in meas:
            iv_list.append(column)
    return iv_list

def dummy_out(data):
    '''replaces NAs with 0, and makes a dummy varialble if it did'''
    for col in data._get_numeric_data().columns:
        if data[col].mean() != data[col].fillna(-1).mean():
            data[col] = data[col].fillna(-1)
            data[col+'_NA'] = (data[col] == -1)
    return data

def drop_unnamed(data):
    '''Drops useless columns created by merge'''
    to_drop = []
    for col in data.columns:
        if 'Unnamed:' in col:
            to_drop.append(col)
    return data.drop(to_drop, 1)

def make_colpro():
    '''Makes the dataset used for analysis by merging precursors'''
    col = pd.read_csv(COL_PATH, engine='c')
    col = col.sort('year')
    col = col.drop_duplicates('Name', 1)
    meas = read_meas()
    col_meas = pd.merge(col, meas, left_on='Name',
                        right_on='name', suffixes=('', '_m'))
    col_meas.to_csv(COL_MEAS_PATH)
    bestrapm = pd.read_csv(BEST_RAPM_PATH)
    colpro = pd.merge(col_meas, bestrapm, left_on='Name',
                      right_on='Name', suffixes=('', '_p'))

    return drop_unnamed(colpro)


def year_fix(year):
    '''Dummies out year. Not Called at the moment'''
    year = str(year).replace(' ', '')
    try:
        return float(year)
    except ValueError:
        print 'ValueError', year
        return 3000

def read_meas(): #TODO WHY THE VALUE ERRORS
    '''Fixes year column. May do other things in future'''
    df = pd.read_def median_fill(data):
    '''replaces NAs with their column's median'''
    median_features = data._get_numeric_data().dropna().median()
    return data._get_numeric_data().fillna(median_features)

csv(MEAS_PATH)
    return df


