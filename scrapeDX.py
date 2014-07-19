import requests
import os
import os.path
import html5lib
import bs4
import numpy as np
import pandas as pd
    
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

def get_rapm(year):
    rapm = pd.read_html('http://stats-for-the-nba.appspot.com/ratings/'+year+'.html',header=0)
    ret = rapm[0]
    ret['year'] = float(year)
    return ret

####MEASUREMENTS###
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

def get_meas():
    url = 'http://www.draftexpress.com/nba-pre-draft-measurements/?page=&year=All&source=All&sort2=ASC&draft=0&pos=0&sort='
    dfs = pd.read_html(url,header = 0)
    df = dfs[5]
    df['name']=df['Name'].str.split(' -').str.get(0)
    df['year']=df['Name'].str.split('-').str.get(1)
    df = df.drop_duplicates(cols='name',take_last=True)
    df = df[:2589]#TODO:WHY
    return df

