import numpy as np
import pandas as pd

def lh_vars(colpro, col_path = 'colData.csv', meas_path = 'measurements.csv'): #TODO: CACHE
    col = drop_unnamed(pd.read_csv(col_path))
    meas = drop_unnamed(pd.read_csv(meas_path))
    cp = drop_unnamed(colpro)
    iv_list = []
    for col in cp._get_numeric_data().columns:
        if col in df._get_numeric_data():
            iv_list.append(col)
    return iv_list

def dummy_out(df):
    for col in df._get_numeric_data().columns:
            if df[col].mean() != df[col].fillna(-1).mean():
                df[col] = df[col].fillna(-1)
                df[col+'_NA'] = (df[col] == -1)
    return df

def median_fill(df):
    median_features = df._get_numeric_data().dropna().median()
    return df._get_numeric_data().fillna(median_features)

def drop_unnamed(df):
    to_drop = []
    for col in df.columns:
        if 'Unnamed:' in col:
            to_drop.append(col)
    return df.drop(to_drop,1)

def add_dummies(combos):
    return_cols = combos
    for x in combos:
        return_cols.append(x + '_NA')
    return return_cols

def make_colpro():
    col = pd.read_csv('colData.csv',engine='c')
    col = col.sort('year')
    col = col.drop_duplicates('Name',1)
    meas = read_meas()
    col_meas = pd.merge(col, meas, left_on ='Name', right_on='name', suffixes=('','_m'))
    col_meas.to_csv('colmeas.csv')
    bestrapm = pd.read_csv('bestRapm.csv')
    colpro = pd.merge(col_meas, bestrapm, left_on='Name',right_on='name', suffixes=('','_p'))
    return drop_unnamed(colpro)

def de_dup(df):
    df = df.sort('year')
    df = df.drop_duplicates('Name',1)
    return df

def year_fix(df):
    df['year'] = df['year'].str.replace(' ','')
    years = np.array(df.year)
    floats = []
    for x in years:
        try:
            floats.append(float(x))
        except:
            floats.append(float(-1))
    df['year'] = floats
    return df

def read_meas():
    read_in = pd.read_csv('measurements.csv')
    df = year_fix(read_in)
    df = df[df.year.astype('float') > 2000]
    return df
