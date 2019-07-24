# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:38:25 2018

@author: BBR
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def iniset():
    datapath='./Data/'
    folder_name=['PRE']
    ele_name=['pre']
    stationid_name='海河流域站点.txt'
    return datapath,folder_name,ele_name,stationid_name

def stationidread(datapath,stationid_name):
    stationid=pd.read_csv(datapath+stationid_name,header=0)
    return stationid

def txtread(datapath,folder_name,txtname):
    filename=datapath+folder_name+'/'+txtname
    txtdata=pd.read_csv(filename,sep='\s+',header=None)
    txtdata.columns=['stationid','lat','lon','alti','year','mon','day',\
                    '20-8','8-20','20-20','20-8c','8-20c','20-20c' ]
    return txtdata

def formatchange(txtdata):
    latnum=txtdata['lat']/100
    latint=latnum.apply(int)
    lonnum=txtdata['lon']/100
    lonint=lonnum.apply(int)
    txtdata['lat']=latint+(txtdata['lat']-\
           latint*100)/60.0
    txtdata['lon']=lonint+(txtdata['lon']-\
          lonint*100)/60.0
    txtdata['alti']=txtdata['alti']/10
    s1=txtdata.loc[:,['20-8','8-20','20-20']]
    s1[s1>30000]=np.nan
    txtdata.loc[:,['20-8','8-20','20-20']]=s1/10
    return txtdata


def data2sql(pdalldata,tablename):
    engine=\
    create_engine\
    ('sqlite:///./Data/pre.db')
    pdalldata.to_sql(tablename,engine,if_exists='append')

startmon='2016-01'
endmon='2018-11'
p=pd.date_range(startmon,endmon,freq='M')
pstr=p.strftime('%Y%m')

for yrmon in pstr:
    txtname='SURF_CLI_CHN_MUL_DAY-PRE-13011-'+yrmon+'.TXT'
    datapath,folder_name,ele_name,stationid_name=iniset()
    txtdata=txtread(datapath,folder_name[0],txtname)
    txtdata=formatchange(txtdata)
    data2sql(txtdata,'obs_pre_daily')
    print(txtname+' was saved to sql')




