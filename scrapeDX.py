import pandas as pd
import requests
import os
import os.path
#import html5lib
#import bs4
def col_stats_scrape(year):#CALL MERGE_PREP TO DO TWO STEPS IN ONE CALL """Usage: col_stats_scrape('2014') Roughly 2 mins per year."""
    front = 'http://www.draftexpress.com/stats.php?sort=8&q='
    frontb ='&league=NCAA&year=20'
    midA = '&league=NCAA&per=per40pace&qual=prospects&sort2=DESC&pos=all&stage=all&min=10&conference=All&pageno='
    back = '&sort=8'
    url = front + frontb + year + midA+ '0' + back

    eff_url = front + 'eff' + frontb + year + midA+ '0' + back
    reg=pd.read_html(url,header=0)[5]
    eff_temps = pd.read_html(eff_url)
    eff_temps[5].to_csv('eff_temp'+year+'.csv')
    eff=pd.read_csv('eff_temp'+year+'.csv',header=3)
    for n in range(1,12):
        url = front + frontb + year + midA+ str(n) + back
        eff_url = front + 'eff'+ frontb + year + midA+ str(n) + back
        reg_temp = pd.read_html(url,header=0)[5]
        eff_temp = pd.read_html(eff_url)[5]
        eff_temp.to_csv('eff_temp'+year+'.csv')
        eff_temp = pd.read_csv('eff_temp'+year+'.csv',header=3)
        reg=reg.append(reg_temp)
        eff=eff.append(eff_temp)
    reg['year']=2000+float(year)
    reg['name']=strip(reg.Name)
    reg=reg.set_index(reg.name)

    eff['year']=2000+float(year)
    eff['name']=strip(eff.Name)
    eff=eff.set_index(eff.name)
    eff.to_csv('effdirty'+year+'.csv')
    df=reg.merge(eff,how='inner',on='name')
    print df.shape
    return df

def pro_stats(year):
    front = 'http://www.draftexpress.com/stats.php?sort=8&q='
    frontb ='&league=NBA&year=20'
    midA = '&per=per40pace&qual=prospects&sort2=DESC&pos=all&stage=&min=10&conference=&pageno='
    back = '&sort=8'
    url = front + frontb + year + midA+ '0' + back
    eff_url = front + 'eff' + frontb + year + midA+ '0' + back
    reg_temps = pd.read_html(url)
    reg_temp = reg_temps[5]
    reg_temp.to_csv('reg_temp'+year+'.csv')
    reg=pd.read_csv('reg_temp'+year+'.csv',header=1)
    eff_temps = pd.read_html(eff_url)
    eff_temps[5].to_csv('eff_temp'+year+'.csv')
    eff=pd.read_csv('eff_temp'+year+'.csv',header=3)
    n = 1 
    url = front + frontb + year + midA+ str(n) + back
    eff_url = front + 'eff'+ frontb + year + midA+ str(n) + back
    reg_temps = pd.read_html(url)
    reg_temp = reg_temps[len(reg_temps)-1]
    reg_temp.to_csv('reg_temp'+year+'.csv')
    reg_temp=pd.read_csv('reg_temp'+year+'.csv',header=1)
    eff_temp = pd.read_html(eff_url)[len(reg_temps)-1]
    eff_temp.to_csv('eff_temp'+year+'.csv')
    eff_temp=pd.read_csv('eff_temp'+year+'.csv',header=3)
    reg=reg.append(reg_temp)
    eff=eff.append(eff_temp)
    reg['year']=2000+float(year)
    reg['name']=strip(reg.Name)
    reg=reg.set_index(reg.name)
    eff['year']=2000+float(year)
    eff['name']=strip(eff.Name)
    eff=eff.set_index(eff.name)
    df=reg.merge(eff,how='inner',on='name')
    df.to_csv('pro'+year+'.csv')
    return df

def get_pros():
    df=pro_stats('02')
    df= df.append(pro_stats('03'))
    df= df.append(pro_stats('04'))
    df= df.append(pro_stats('05'))
    df= df.append(pro_stats('06'))
    df= df.append(pro_stats('07'))
    df= df.append(pro_stats('08'))
    df= df.append(pro_stats('09'))
    df= df.append(pro_stats('10'))
    df= df.append(pro_stats('11'))
    df= df.append(pro_stats('12'))
    df= df.append(pro_stats('13'))
    df= df.append(pro_stats('14'))
    df.to_csv('proData.csv')
    print df.shape
    return df

def get_rapm(year):
    rapm = pd.read_html('http://stats-for-the-nba.appspot.com/ratings/'+year+'.html',header=0)
    ret = rapm[0]
    ret['year'] = year
    return ret

def get_all_rapm():
    a = get_rapm('2001')
    b = get_rapm('2002')
    c = get_rapm('2003')
    d = get_rapm('2004')
    e = get_rapm('2005')
    f = get_rapm('2006')
    g = get_rapm('2007')
    h = get_rapm('2008')
    i = get_rapm('2009')
    j = get_rapm('2010')
    k = get_rapm('2011')
    l = get_rapm('2012')
    m = get_rapm('2013')
    df =  pd.concat([a,b,c,d,e,f,g,h,i,j,k,l,m])
    df = df.drop('Unnamed: 5' , 1)
    df['name']=df.Name
    df = df.drop('Name',1)
    df = df.rename(columns={'Defense per 100':'def100', 'Off+Def per 200': 'tot200',\
            'Offense per 100':'off100'})
    return df

def get_cols():
    df=col_scrape('02')
    df= df.append(col_scrape('03'))
    df= df.append(col_scrape('04'))
    df= df.append(col_scrape('05'))
    df= df.append(col_scrape('06'))
    df= df.append(col_scrape('07'))
    df= df.append(col_scrape('08'))
    df= df.append(col_scrape('09'))
    df= df.append(col_scrape('10'))
    df= df.append(col_scrape('11'))
    df= df.append(col_scrape('12'))
    df= df.append(col_scrape('13'))
    df= df.append(col_scrape('14'))
    df.to_csv('colData.csv')
    print df.shape
    return df

def stripquote(text):
	if (text[-1] == '"'):
		return text[0:-1]
	else:
		return text

def heightfix(df,flen):
	df.new=df.str.split(' ')
	df.feet=df.new.str[0]
	df.feet=df.feet.str[flen].astype('float')
	df.inches=df.new.str[1]
	df.inches=df.inches.str[0:-1]
	df.inches=df.inches.astype('float')
	df.old=df.new.replace('"','',regex=True)
	df.final= (12*df.feet+df.inches).astype('float')
	return df.final

def strip(text):
	try:
		return text.strip()
	except AttributeError:
		return text

def meas():
	url='http://www.draftexpress.com/nba-pre-draft-measurements/?page=&year=All&source=All&sort2=ASC&draft=0&pos=0&sort='
	df= pd.read_csv('dirtyMeas.csv')
	df['name']=df['Name'].str.split(' -').str.get(0)
	df['year']=df['Name'].str.split('-').str.get(1)
	df=df.drop_duplicates(cols='name',take_last=True)
	df=df[0:2589]
	df=df.drop('Name',1)
	df=df.drop('Rank',1)
	#FIX HEIGHTS
	df['heightshoes']=heightfix(df['Height w/shoes'],0)
	df=df.drop('Height w/shoes',1)
	df['heightbare']=heightfix(df['Height w/o Shoes'],0)
	df=df.drop('Height w/o Shoes',1)
	df['wingspan']=heightfix(df['Wingspan'],0)
	df=df.drop('Wingspan',1)
	df['reach']=heightfix(df['Reach'],0)
	#TODO: FIX standvert reach and maxvertreach
	df['standvertreach']=heightfix(df['No Step Vert Reach'],0)
	df['maxvertreach']=heightfix(df['Max Vert Reach'],0)
	df = df.set_index(df.name)
	df=df.sort()
	df.to_csv('measurements.csv')
	return df

def col_merge(year):
	pd.options.display.max_columns = 200
	meas = pd.read_csv('measurements.csv')
	col= col_scrape(year)
	df=pd.merge(col,meas,on='name',how='left')
	df=df.drop('Cmp',1)
	df=df.drop('Name',1)
	df=df.drop_duplicates(cols='name',take_last=True)
	df.to_csv('mergedCol'+year+'.csv')

	return df


