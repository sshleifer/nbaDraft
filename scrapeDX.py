import pandas as pd
import requests
import os
import os.path
import html5lib
import bs4
import numpy as np
import statsmodels.api as sm
from itertools import combinations
#sm.RLM(Y,X, M=sm.robust.norms.HuberT())

def main():
    #get_all_cols()
    col = pd.read_csv('colData.csv',engine='c')
    #get_all_pros()
    pro = pd.read_csv('proData.csv')
    pro = pro.drop('Unnamed: 0',1)
    #rapm = get_all_rapm()
    bestrapm = pd.read_csv('bestRapm.csv')
    colpro = pd.merge(col, bestrapm, left_on='Name',right_on='name', suffixes=('','_p'))
    regress(colpro[lh_list()],colpro['off100'])    
    #genetic_loop(colpro[lh_list()],colpro['off100'])
    return dummy_out(colpro) 

def genetic_loop(ivs, colpro):
    best_yet = 100000
    temp = (1,1)
    for i, combo in enumerate(combinations(ivs,10)):
        result = regress(colpro[list(combo)], colpro['off100'])
        if result.aic < best_yet:
            temp = (result, combo)
    return temp

def drop_unnamed(df):
    to_drop = []
    for col in df.columns:
        if 'Unnamed:' in col:
            to_drop.append(col)
    return df.drop(to_drop,1)

def regress(X, Y):
    results = sm.OLS(Y,X)
    results = results.fit()
    return results

def lh_list(colpro): #TODO: CACHE
    df = drop_unnamed(pd.read_csv('colData.csv'))
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


def get_stats(year,level='pro'):
    front = 'http://www.draftexpress.com/stats.php?sort=8&q='
    pages = 2
    frontb ='&league=NBA&year=20'
    if level == 'col':
        frontb ='&league=NCAA&year=20'
        pages = 13
    midA = '&per=per40pace&qual=prospects&sort2=DESC&pos=all&stage=all&min=10&conference=All&pageno='
    back = '&sort=8'
    url = front + frontb + year + midA+ '0' + back
    reg = pd.DataFrame()
    eff = pd.DataFrame()
    for n in xrange(pages):
        url = front + frontb + year + midA+ str(n) + back
        eff_url = front + 'eff'+ frontb + year + midA+ str(n) + back
        reg_temps = pd.read_html(url,header=0)
        reg_temp = reg_temps[5]
        eff_temps = pd.read_html(eff_url)
        eff_temp = eff_temps[5]
        eff_temp.to_csv('eff_temp.csv')
        eff_temp = pd.read_csv('eff_temp.csv',header=3)
        reg = reg.append(reg_temp)
        eff = eff.append(eff_temp)
    reg['year']=2000+float(year)
    eff['year']=2000+float(year)
    df=reg.merge(eff,how='inner',on='Name', suffixes=('','_y'))
    df = df.drop(['Cmp','Team_y','year_y','Min_y','Cmp_y','GP_y'],1)
    print df.shape
    #df.to_csv(level+ '_stats/' + year + '.csv')
    return df

def get_all_cols(): #18minutes Produces colData.csv
    df = get_stats('02')
    df= df.append(get_stats('03','col'))
    df= df.append(get_stats('04','col'))
    df= df.append(get_stats('05', 'col'))
    df= df.append(get_stats('06', 'col'))
    df= df.append(get_stats('07', 'col'))
    df= df.append(get_stats('08', 'col'))
    df= df.append(get_stats('09', 'col'))
    df= df.append(get_stats('10', 'col'))
    df= df.append(get_stats('11', 'col'))
    df= df.append(get_stats('12', 'col'))
    df= df.append(get_stats('13', 'col'))
    df= df.append(get_stats('14', 'col'))
    df.to_csv('colData.csv')
    print df.shape
    return df

def get_all_pros():#Produces proData.csv
    df = get_stats('02')
    df= df.append(get_stats('03'))
    df= df.append(get_stats('04'))
    df= df.append(get_stats('05'))
    df= df.append(get_stats('06'))
    df= df.append(get_stats('07'))
    df= df.append(get_stats('08'))
    df= df.append(get_stats('09'))
    df= df.append(get_stats('10'))
    df= df.append(get_stats('11'))
    df= df.append(get_stats('12'))
    df= df.append(get_stats('13'))
    df= df.append(get_stats('14'))
    df.to_csv('proData.csv')
    print df.shape
    return df

def get_rapm(year):
    rapm = pd.read_html('http://stats-for-the-nba.appspot.com/ratings/'+year+'.html',header=0)
    ret = rapm[0]
    ret['year'] = float(year)
    return ret

def get_all_rapm(): #Produced bestRapm.csv
    df = pd.concat([get_rapm('2001'),get_rapm('2002'), get_rapm('2003')\
            , get_rapm('2004'), get_rapm('2005'), get_rapm('2006')\
            , get_rapm('2007'), get_rapm('2008'), get_rapm('2009'), get_rapm('2010'), get_rapm('2011')\
            , get_rapm('2012'),get_rapm('2013')])
    df = df.drop('Unnamed: 5' , 1)
    df['name']=df.Name
    df = df.drop('Name',1)
    df = df.rename(columns={'Defense per 100':'def100', 'Off+Def per 200': 'tot200',\
            'Offense per 100':'off100'})
    rapm = df
    rapm.to_csv('allRapm.csv')
    rapm = rapm.sort('tot200', ascending=True)
    rapm = rapm.drop_duplicates('name',take_last=True)
    rapm.to_csv('bestRapm.csv')
    return rapm

def height_fix(df,flen):
    df.new=df.str.split(' ')
    df.feet=df.new.str[0]
    df.feet=df.feet.str[flen].astype('float')
    df.inches=df.new.str[1]
    df.inches=df.inches.str[0:-1]
    df.inches=df.inches.astype('float')
    df.old=df.new.replace('"','',regex=True)
    df.final= (12*df.feet+df.inches).astype('float')
    return df.final

def meas_full():
    df = meas()
    return clean_meas(df)

def meas():
    url = 'http://www.draftexpress.com/nba-pre-draft-measurements/?page=&year=All&source=All&sort2=ASC&draft=0&pos=0&sort='
    dfs = pd.read_html(url,header = 0)
    df = dfs[5]
    df['name']=df['Name'].str.split(' -').str.get(0)
    df['year']=df['Name'].str.split('-').str.get(1)
    df = df.drop_duplicates(cols='name',take_last=True)
    df = df[:2589]#TODO:WHY
    return df

def clean_meas(df):
    df['heightshoes'] = height_fix(df['Height w/shoes'],0)
    df['heightbare'] = height_fix(df['Height w/o Shoes'],0)
    df['wingspan'] = height_fix(df['Wingspan'],0)
    df['reach'] = height_fix(df['Reach'],0)
    #TODO: FIX standvert reach and maxvertreach
    df['standvertreach'] = height_fix(df['No Step Vert Reach'],0)
    df['maxvertreach'] = height_fix(df['Max Vert Reach'],0)
    df = df.drop(['Name','Rank','Height w/shoes', 'Height w/o Shoes','Wingspan', 'No Step Vert Reach', 'Max Vert Reach'],1)
    df = df.sort()
    df.to_csv('measuremeets.csv', encoding = 'utf-8')
    return df

def col_merge(year):
    pd.options.display.max_columns = 200
    meas = pd.read_csv('measurements.csv')
    col= get_stats(year)
    df=pd.merge(col,meas,on='name',how='left')
    df=df.drop('Cmp',1)
    df=df.drop('Name',1)
    df=df.drop_duplicates(cols='name',take_last=True)
    df.to_csv('mergedCol'+year+'.csv')
    return df
