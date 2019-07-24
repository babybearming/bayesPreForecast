# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:25:01 2019

@author: babybearming
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def iniset():
    datapath='./Data/'
    folder_name=['PRE']
    ele_name=['pre']
    stationid_name='station_255.txt'
    return datapath,folder_name,ele_name,stationid_name

def stationidread(datapath,stationid_name):
    stationid=pd.read_csv(datapath+stationid_name,sep=' ',header=0)
    return stationid

def sql2data(myQuery):
    engine=\
    create_engine\
    ('sqlite:///./Data/pre.db')
    df=pd.read_sql_query(myQuery,engine)
    return df 

def readstadata(stationid,year):
    myquery='''SELECT stationid,year,mon,day,"20-20" FROM obs_pre_daily \
    where stationid={} and year={} and mon>=6 and mon<=10'''.format(stationid,year)
    df=sql2data(myquery)
    return df 

datapath,folder_name,ele_name,stationid_name=iniset()
stationid=stationidread(datapath,stationid_name)
stanums=len(stationid['id'])

startyr=2017
endyr=2018
for yr in range(startyr,endyr+1):
    print(yr)
    yrstr=str(yr)
    stadata=[i for i in range(stanums)]
    for inx,sta in enumerate(stationid['id']):
        print(inx,sta)
        stadata[inx]=readstadata(sta,yr)
    datenums=len(stadata[0])
    stadataall=np.full((datenums,stanums),np.nan)
    for inx in range(stanums):
        stadataall[:,inx]=stadata[inx]["20-20"] 
    np.save('./Data/obsstadata/'+yrstr+'.npy',stadataall)