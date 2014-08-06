import pandas as pd
#import numpy as np
from constants import DX_YEARS

def get_stats(year, level='pro'): #TODO Switch to regex patterns
    '''Scrapes draftexpress.com/stats for of a given level, year'''
    front = 'http://www.draftexpress.com/stats.php?sort=8&q='
    pages = 2
    frontb = '&league=NBA&year=20'
    if level == 'col':
        frontb = '&league=NCAA&year=20'
        pages = 13
    midA = '&per=per40pace&qual=prospects&sort2=DESC&pos=all&stage=all&min=10&conference=All&pageno='
    back = '&sort=8'
    url = front + frontb + year + midA+ '0' + back
    reg = pd.DataFrame()
    eff = pd.DataFrame()
    for n in xrange(pages):
        url = front + frontb + year + midA+ str(n) + back
        eff_url = front + 'eff'+ frontb + year + midA+ str(n) + back
        reg_temps = pd.read_html(url, header=0)
        reg_temp = reg_temps[5]
        eff_temps = pd.read_html(eff_url)
        eff_temp = eff_temps[5]
        eff_temp.to_csv('temp.csv')
        eff_temp = pd.read_csv('temp.csv', header=3) #im ashamed
        reg = reg.append(reg_temp)
        eff = eff.append(eff_temp)
    reg['year'] = 2000 + float(year)
    eff['year'] = 2000 + float(year)
    df = reg.merge(eff, how='inner', on='Name', suffixes=('', '_y'))
    df = df.drop(['Cmp', 'Team_y', 'year_y', 'Min_y', 'Cmp_y', 'GP_y'], 1)
    print df.shape
    return df

def get_all_cols(): #Produces colData.csv in 18 minutes
    data = pd.DataFrame()
    for year in DX_YEARS:
        print year
        data.append(get_stats(year, 'col'))
    data.to_csv('datasets/colData.csv')
    print data.shape
    return data

def get_all_pros():#Produces proData.csv
    data = pd.DataFrame()
    for year in DX_YEARS:
        print year
        data.append(get_stats(year, 'pro'))
    data.to_csv('datasets/proData.csv')
    print data.shape
    return data
