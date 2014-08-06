from constants import RAPM_URL, MEAS_URL
import pandas as pd
###RAPM STATS###
def get_all_rapm(): #Produced bestRapm.csv
    rapm = pd.concat([get_rapm('2001'), get_rapm('2002'), get_rapm('2003'),
                      get_rapm('2004'), get_rapm('2005'), get_rapm('2006'),
                      get_rapm('2007'), get_rapm('2008'), get_rapm('2009'),
                      get_rapm('2010'), get_rapm('2011'), get_rapm('2012'),
                      get_rapm('2013')])
    rapm = rapm.rename(columns={'Defense per 100':'def100',
                                'Off+Def per 200': 'tot200',
                                'Offense per 100':'off100'})
    rapm = rapm.drop(['Unnamed: 5'], 1)
    rapm.to_csv('datasets/allRapm.csv')
    rapm = rapm.sort('tot200', ascending=True)
    rapm = rapm.drop_duplicates('Name', take_last=True)
    rapm.to_csv('datasets/bestRapm.csv')
    return rapm

def get_rapm(year):
    url = RAPM_URL + year + '.html'
    rapm = pd.read_html(url, header=0)
    ret = rapm[0]
    ret['year'] = float(year)
    return ret

####MEASUREMENTS###
def get_meas():
    dfs = pd.read_html(MEAS_URL, header=0)
    df = dfs[5]
    dQf['name'] = df['Name'].str.split(' -').str.get(0)
    df['year'] = df['Name'].str.split('-').str.get(1)
    df = df.drop_duplicates(cols='name', take_last=True)
    df = df[:2589]#TODO:WHY
    return df
