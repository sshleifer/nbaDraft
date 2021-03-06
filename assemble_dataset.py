'''Assembles and preprocesses the colPro.csv, the dataset used for analysis'''
import numpy as np
import pandas as pd
MEAS_PATH = 'datasets/measurements.csv'
COL_PATH = 'datasets/colData.csv'
BEST_RAPM_PATH = 'datasets/bestRapm.csv'
COL_MEAS_PATH = 'datasets/colmeas.csv'
DRAFT_PATH = 'datasets/draft_mix.csv'

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
    col_meas.to_csv('datasets/colmeas.csv')
    col_meas = add_draft(col_meas)
    col_meas = col_meas.drop(['weight'], 1)
    x_test = col_meas[col_meas.year_d==2014]

    col_meas['pick'] = col_meas['pick'].fillna(65)
    bestrapm = pd.read_csv('datasets/bestRapm.csv')

    colpro = pd.merge(col_meas, bestrapm, left_on='Name',
                      right_on='Name', suffixes=('', '_p'))
    colpro = small_clean(colpro)
   # colpro = add_draft(colpro)
    return drop_unnamed(colpro), drop_unnamed(x_test)

def small_clean(colpro):
    colpro['Hand Length'] = colpro['Hand Length'].apply(lambda x: fake_zero(x))
    colpro['Hand Width'] = colpro['Hand Width'].apply(lambda x: fake_zero(x))
    return colpro

def fake_zero(x):
    if x == 0:
        return None
    else:
        return x

def add_draft(colpro, how='left'):
    draft = pd.read_csv(DRAFT_PATH)
    new = pd.merge(colpro, draft, on='Name', how=how, suffixes=('', '_d'))
    df = use_draft(new)
    df = df.drop(['weight','ac_year'],1)
    df['pick'] = df.pick.fillna(65)
    return new

def use_draft(df):
    df['heightshoes'] = fill_with_vals(np.array(df.heightshoes), np.array(df.height))
    df['heightbare'] = fill_with_vals(np.array(df.heightbare), np.array(df.height))
    df['heightbare'] = fill_with_vals(np.array(df.heightbare), np.array(df.heightshoes))
    df['heightshoes'] = fill_with_vals(np.array(df.heightshoes), np.array(df.heightbare))
    df['Weight'] = fill_with_vals(np.array(df.Weight), np.array(df.weight))
    return df

def fill_with_vals(a, b):
    assert len(a) == len(b), 'COLUMNS ARE NOT OF EQUAL LENGTH'
    for i, value in enumerate(a):
        if value > 0:
            pass
        else:
            if b[i]:
                a[i] = b[i]
    return a
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
    df = pd.read_csv(MEAS_PATH)
    return df
