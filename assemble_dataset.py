import numpy as np
import pandas as pd
import statsmodels.api as sm

def lh_vars(colpro): #TODO: CACHE
    df = drop_unnamed(pd.read_csv('colmeas.csv'))
    cp = dummy_out(drop_unnamed(colpro))
    df = dummy_out(df)
    df = drop_unnamed(df)
    iv_list = []
    for col in df._get_numeric_data().columns:
        if col in cp._get_numeric_data():
            iv_list.append(col)
    return iv_list

def dummy_out(df):
    for col in df._get_numeric_data().columns:
            if df[col].mean() != df[col].fillna(-1).mean():
                df[col] = df[col].fillna(-1)
                df[col+'_NA'] = (df[col] == -1)
    return df

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
    return dummy_out(colpro)

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
    df = dummy_out(df)
    df = df[df.year.astype('float') > 2000]
    return df
