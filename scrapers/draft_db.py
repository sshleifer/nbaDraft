'''Scrapes data from DraftExpress.com to produce datasets used for prediction'''
import html5lib
import bs4
import numpy as np
import pandas as pd
from constants import MOCK_URL, MEAS_URL, RAPM_URL, DX_YEARS, DRAFT_URL
from assemble_dataset import drop_unnamed, order
DRAFT_YEARS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
MOCK_PATH = 'datasets/dx_mock_drafts.csv'
DRAFT_PATH = 'datasets/drafts.csv'
DRAFT_MIX_PATH = 'datasets/draft_mix.csv'
DRAFT_ORDER = ['year', 'Name', 'school', 'height', 'weight', 'pick_est', 'ac_year', 'from_mock']
###MOCK DRAFT AND REAL DRAFT STATS###
def scrape_draft(year):
    def age_from_bday(x):
        try:
            return year - int (x[:4])
        except:
            return None

    url = DRAFT_URL + str(year)
    dfs = pd.read_html(url, header=0)
    draft = dfs[-1][['Pick','Name', 'Birthday']]
    draft['pick'] = draft.index + 1
    draft['age'] = draft['Birthday'].astype('string').apply(age_from_bday)
    draft['year'] = year
    return draft[['Name','pick', 'age', 'year']]

def draft_db():
    db = pd.DataFrame()
    for year in DRAFT_YEARS:
        print year
        db = db.append(scrape_draft(year))
        print db.year
    db.to_csv(DRAFT_PATH)
    return db

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

def gather_picks():
    mocks = pd.read_csv(MOCK_PATH)
    drafts = pd.read_csv(DRAFT_PATH)
    new = pd.merge(drafts, mocks, left_on='Name', right_on='name', how='left',suffixes=('', '_M'))
    new['pick_est'] = new.pick_M.fillna(new.pick)
    new['year'] = new.year.fillna(new.year_M).astype('int')
    new['from_mock'] = new.year >= 2007
    new = drop_unnamed(new.drop(['name','pick', 'pick.1','pick_M', 'year_M', 'age_M'], 1)).sort('year', ascending=False)
    new = order(new[DRAFT_ORDER],DRAFT_ORDER)
    new.to_csv(DRAFT_MIX_PATH)
    return new

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
    m_draft['s'] = m_draft['s'].apply(wierd_name_fix)
    m_draft['name'] = m_draft['s'].apply(lambda x: x[0] + ' ' + x[1])
    m_draft['pos'] = m_draft['s'].apply(lambda x: untangle_pos(x[2]))
    m_draft['age'] = m_draft['s'].apply(lambda x: int(x[2][-2:])- offset)
    m_draft['height'] = m_draft['s'].apply(lambda x: height_to_inch(x[4][:-2]))
    m_draft['weight'] = m_draft['s'].apply(lambda x: int(x[5]))
    m_draft['school'] = m_draft['s'].apply(untangle_school)
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
