import pandas as pd
#import html5lib
import requests
#from bs4 import BeautifulSoup
import os
import os.path

def col_scrape(year):#CALL MERGE_PREP TO DO TWO STEPS IN ONE CALL
    #Rougly 2 minutes per year
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
        eff_temp=pd.read_csv('eff_temp'+year+'.csv',header=3)
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
def nth(url,n):
	dfs = pd.read_html(url,header=0)
	ret  = dfs[n]
	print "Shape = ",ret.shape
	return ret

def col_merge(year):
	import pandas as pd
	pd.options.display.max_columns = 200
	meas = pd.read_csv('measurements.csv')
	col= col_scrape(year)
	df=pd.merge(col,meas,on='name',how='left')
	df=df.drop('Cmp',1)
	df=df.drop('Name',1)
	df=df.drop_duplicates(cols='name',take_last=True)
	df.to_csv('mergedCol'+year+'.csv')

	return df
def col_get():
	col_merge('03')
	col_merge('04')
	col_merge('05')
	col_merge('06')
	col_merge('07')
	col_merge('08')
	col_merge('09')
	col_merge('10')
	col_merge('11')
	col_merge('12')
	col_merge('13')

def pro_merge(seasA,seasB,seasC):
	pathA='c'+seasA+'.csv'
	pathB='c'+seasB+'.csv'
	pathC='c'+seasC+'.csv'
	proA=pd.io.parsers.read_csv('cleanESPN/'+pathA,header=0)
	proB=pd.io.parsers.read_csv('cleanESPN/'+pathB,header=0)
	proC=pd.io.parsers.read_csv('cleanESPN/'+pathC,header=0)
	proA=proA.set_index(proA.name)
	print proA.head()
	proB=proB.set_index(proB.name)
	proC=proC.set_index(proC.name)
	print proA.shape, proB.shape, proC.shape
	df=pd.merge(proA,proB,on='name',how='outer',suffixes=('','_B'))
	new=pd.merge(df,proC,on='name',how='outer',suffixes=('','_C'))
	new.to_csv('ThreePro/three'+seasA+'.csv',encoding='utf-8')
	print "becomes",new.shape
	return proA,proB,proC,new
def many_merge():
	new=pro_merge('2003','2004','2005')
	pro_merge('2004','2005','2006')
	pro_merge('2005','2006','2007')
	pro_merge('2006','2007','2008')
	pro_merge('2007','2008','2009')
	pro_merge('2008','2009','2010')
	pro_merge('2009','2010','2011')
	pro_merge('2010','2011','2012')
	pro_merge('2011','2012','2013')
	pro_merge('2012','2013','2014')
	return new
def espn_scrape(year):
	front ='http://insider.espn.go.com/nba/hollinger/statistics/_/sort/VORPe/'
	back ='/year/'+ year+'/qualified/false'
	dfs = pd.read_html(front+back,header=1)
	df =dfs[0]
	for n in range(1,11):
		url = front +'page/' + str(n) + back
		dfs = pd.read_html(url,header=1)
		temp= dfs[0]
		df=df.append(temp)
	path ='ESPN/'+ year +'.csv'
	df.to_csv(path, sep=',', encoding='utf-8')
	return df
def pro_col(year,seasA):
	path ='ThreePro/three'+seasA+'.csv'
	pro=pd.io.parsers.read_csv(path,header=0)
	pathB='mergedCol/mergedCol'+year+'.csv'
	col=pd.io.parsers.read_csv(pathB,header=0)
	df=pd.merge(col,pro,on='name',how='left',suffixes=('_COL',''))
	df.G=df.GP.fillna(0,inplace=True)
	df=df.sort(df.G)
	df=df.set_index(df.name)
	df.to_csv('finalDB/' + year + '.csv')	
	return df
def pro_coller():
	a=pro_col('03','2004')
	b=pro_col('04','2005')
	c=pro_col('05','2006')
	d=pro_col('06','2007')
	e=pro_col('07','2008')
	f=pro_col('08','2009')
	g=pro_col('09','2010')
	h=pro_col('10','2011')
	
	df=pd.concat([pd.concat([pd.concat([a,b]),pd.concat([c,d])]),pd.concat([pd.concat([e,f]),pd.concat([g,h])])])
	return df




def stuffer_clean(seas):
	import pandas
	new=pd.io.parsers.read_csv('STUFFER/NBA'+seas+'.csv',header=0)
	new=new.rename(columns=giveNames(seas),index={'POS':'AGE'})
	new=new.drop_duplicates(cols='name',take_last=True)	
	new = new.drop('unnamed',1)
	new=new.set_index(new.name)
	new=new.sort()
	new.to_csv('cleanNBA'+ seas+ '.csv')
	return new
def giveNames(seas):
	newNames={}
	if (seas == '2009-2010') :
		print 'got here',seas
		old= ['Unnamed: 0', seas + 'NBA REGULAR SEASON PLAYER STATS', 'Rank', '\nPLAYER', 'TEAM', 'TRADE', 'POS', 'AGE', 'GP', 'GM', 'GS', 'ORTG', 'DRTG', 'USG%', 'TOr', '3PA', '3P%', 'FTA', 'FT%', 'eFG%', 'TS%', 'MPG', 'PPG', 'RPG', 'APG', 'VI']
		new= [seas, 'Rank', 'name', 'TEAM', 'TRADE', 'POS', 'AGE', 'GP', 'GM', 'GS', 'ORTG', 'DRTG', 'USG%', 'TOr', '3PA', '3P%', 'FTA', 'FT%', 'eFG%', 'TS%', 'MPG', 'PPG', 'RPG', 'APG', 'VI','unnamed']
		newNames=dict(zip(old,new))
	elif ((seas=='2010-2011') or (seas =='2011-2012')):
		old= ['Unnamed: 0', seas+ ' NBA REGULAR SEASON PLAYER STATS', 'Rank', '\nPLAYER', 'TEAM', 'POS', 'AGE', 'GP', 'MPG', 'MIN%', 'USG%', 'TOr', 'FTA', 'FT%', '2PA', '2P%', '3PA', '3P%', 'TS%', 'PPG', 'RPG', 'TRB%', 'APG', 'AST%', 'VI','SPG', 'BPG']
		new= ['JELLO', 'Rank', 'name', 'TEAM', 'POS', 'AGE', 'GP', 'MPG', 'MIN%', 'USG%', 'TOr', 'FTA', 'FT%', '2PA', '2P%', '3PA', '3P%', 'TS%', 'PPG', 'RPG', 'TRB%', 'APG', 'AST%', 'VI','SPG', 'BPG','unnamed']
		newNames=dict(zip(old,new))
	if ((seas=='2012-2013') or (seas=='2013-2014')):		
		old=['Unnamed: 0', seas + 'NBA REGULAR SEASON PLAYER STATS', 'Rank', '\nPLAYER', 'TEAM', 'POS', 'AGE', 'GP', 'MPG', 'MIN%', 'USG%', 'TOr', 'FTA', 'FT%', '2PA', '2P%', '3PA', '3P%', 'TS%', 'PPG', 'RPG', 'TRB%', 'APG', 'AST%', 'SPG', 'BPG', 'VI']		
		new=[seas, 'Rank', 'name', 'TEAM', 'POS', 'AGE', 'GP', 'MPG', 'MIN%', 'USG%', 'TOr', 'FTA', 'FT%', '2PA', '2P%', '3PA', '3P%', 'TS%', 'PPG', 'RPG', 'TRB%', 'APG', 'AST%', 'SPG', 'BPG', 'VI','unnamed']		
		newNames=dict(zip(old,new))
			
	return newNames
def espn_clean(year):
	new=pd.io.parsers.read_csv('ESPN/'+year+ '.csv',header=0)
	new.nametag=new.PLAYER.str.split(',')
	new['name']=new.nametag.str[0]
	new['team']=strip(new.nametag.str[1])
	new=new.set_index('name')
	#new=new.drop_duplicates(cols='RK',take_last=True)
	new=new.drop('Unnamed: 0',1)
	new=new.drop('PLAYER',1)
	new.to_csv('cleanESPN/c'+ year+ '.csv')
	
	return new
if __name__ == "__main__":
	df=pd.io.parsers.read_csv('finalDB/pasted.csv',header=0)
def db_merge(patha,pathb):
	dfA=pd.io.parsers.read_csv(patha,header=0)
	dfB=pd.io.parsers.read_csv(pathb,header=0)
	pieces=[dfA,dfB]
	print dfA.shape
	print dfB.shape
	"""
	new=pd.concat(pieces,axis=0)
	new=new.set_index(new.name)
	new=new.sort()
	new.to_csv('merged.csv')
	new=new[(new.GP > 0)]
	"""
	return dfA,dfB
def final_merge(list):
	for n in range (0, len(list)-1):
		print n
		path='finalDB/'+list[n]+'.csv'
		pieces[n]=(pieces,pd.read_csv(path,header=0))
	pieces=pd.concat(pieces)
	return pieces
