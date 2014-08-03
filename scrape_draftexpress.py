'''Scrapes data from DraftExpress.com to produce datasets used for prediction'''
import html5lib
import bs4
import numpy as np
import pandas as pd
from constants import MOCK_URL, MEAS_URL, RAPM_URL, DX_YEARS

def scrape_mock(year):
    '''Scrapes a mock_draft off the web in a wierd format'''
    url = MOCK_URL + str(year) + '/list/'
    crap = pd.read_html(url, header=0, match='First Round')
    first_round = crap[-1]
    crap = pd.read_html(url, header=0, match='Second Round')
    second_round = crap[-1]
    first_round.columns = ['pick', 'year', 'details']
    second_round.columns = ['pick', 'year', 'details']
    second_round['pick'] = second_round['pick'] + 30
    mock_draft = first_round.append(second_round)
    mock_draft['year'] = year
    mock_draft = mock_draft.set_index('pick')
    mock_draft['pick'] = mock_draft.index
    return mock_draft


def parse_mock_details(m_draft):
    '''cleans the ugly column of the scraped m_draft'''
    def untangle_pos(x):
        tweeners = ['PG/SG', 'SG/SF', 'SF/PF', 'PF/C']
        for i in tweeners:
            if i in x:
                return i
        pos_list = ['PG', 'SG', 'SF', 'PF', 'C']
        for i in pos_list:
            if i in x:
                return i

    def untangle_school(x):
        first_part = x[6][4:]
        if len(x) == 8:
            return first_part[:-1]
        if len(x) == 9:
            return first_part + x[7][:-1]
        else:
            return first_part

    def height_to_inch(x):
        feet = int(x[0])
        inches = int(x[2:])
        return 12 * feet + inches

    def wierd_name_fix(x):
        if 'Jr' in x[2]:
            x[2:] = x[3:]
        elif x[2] == 'Colo':
            x[1] += x[2]
            x[2:] = x[3:]
        return x

    def years_since(df):
        return 2014 - df.year.mean()

    offset = years_since(m_draft)
    m_draft['s'] = m_draft['details'].str.split(' ')
    m_draft['s'] = m_draft['s'].apply(lambda x: wierd_name_fix(x))
    m_draft['name'] = m_draft['s'].apply(lambda x: x[0] + ' ' + x[1])
    m_draft['pos'] = m_draft['s'].apply(lambda x: untangle_pos(x[2]))
    m_draft['age'] = m_draft['s'].apply(lambda x: int(x[2][-2:])- offset)
    m_draft['height'] = m_draft['s'].apply(lambda x: height_to_inch(x[4][:-2]))
    m_draft['weight'] = m_draft['s'].apply(lambda x: int(x[5]))
    m_draft['school'] = m_draft['s'].apply(lambda x: untangle_school(x))
    m_draft['ac_year'] = m_draft['s'].apply(lambda x: x[-1])
    m_draft = m_draft.drop(['s', 'details'], 1)
    return m_draft


def make_mock_db(): #30 Seconds
    '''Gets 8 mock drafts and appends them into one'''
    years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    mock_db = pd.DataFrame()
    for year in years:
        print year
        mock_db = mock_db.append(parse_mock_details(scrape_mock(year)))
    return mock_db


def get_stats(year, level='pro'): #TODO Switch to regex patterns
    '''Scrapes draftexpress.com/stats for of a given level, year'''
    front = 'http://www.draftexpress.com/stats.php?sort=8&q='
    pages = 2
    frontb = '&league=NBA&year=20'
    if level == 'col':
        frontb = '&league=NCAA&year=20'
        pages = 13
    midA = '&per=per40pace&qual=prospects&sort2=DESC\
            &pos=all&stage=all&min=10&conference=All&pageno='
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
        eff_temp.to_csv('datasets/eff_temp.csv')
        eff_temp = pd.read_csv('datasets/eff_temp.csv', header=3)
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
        data.append(get_stats(year, 'col'))
    data.to_csv('datasets/colData.csv')
    print data.shape
    return data

def get_all_pros():#Produces proData.csv
    data = pd.DataFrame()
    for year in DX_YEARS:
        data.append(get_stats(year, 'pro'))
    data.to_csv('datasets/proData.csv')
    print data.shape
    return data

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
    df['name'] = df['Name'].str.split(' -').str.get(0)
    df['year'] = df['Name'].str.split('-').str.get(1)
    df = df.drop_duplicates(cols='name', take_last=True)
    df = df[:2589]#TODO:WHY
    return df
