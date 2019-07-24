# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:48:36 2019

@author: Administrator
"""
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def iniset():
    datapath='./Data/'
    folder_name=['PRE']
    ele_name=['pre']
    stationid_name='station_255.txt'
    return datapath,folder_name,ele_name,stationid_name

def stationidread(datapath,stationid_name):
    stationid=pd.read_csv(datapath+stationid_name,sep=' ',header=0)
    return stationid

datapath,folder_name,ele_name,stationid_name=iniset()
stationid=stationidread(datapath,stationid_name)
lonlat=stationid[['lon','lat']].values

stanums=255;foredaynums=1;

nowdate=datetime.datetime(2018,7,23)  
results_path='./results/'+nowdate.strftime('%Y%m%d')+'/'
sta_precdf=np.zeros([stanums,6])  #各站小雨、中雨...特大暴雨 的累积概率密度
products_path='./products/data/'+nowdate.strftime('%Y%m%d')+'/'
if os.path.exists(products_path)==False:
    os.makedirs(products_path)

for foreday in range(foredaynums):
    filename_faipmf=results_path+'rsfaipmf_'+str(foreday+1)+'day.npy'
    filename_pypmf=results_path+'rspypmf_'+str(foreday+1)+'day.npy'
    rsfaipmf=np.load(filename_faipmf)
    rspypmf=np.load(filename_pypmf)
    for sta in range(stanums):
        try:
            sta_precdf[sta,0]=np.sum(rsfaipmf[sta][0:9])
            sta_precdf[sta,1]=np.sum(rsfaipmf[sta][9:24])
            sta_precdf[sta,2]=np.sum(rsfaipmf[sta][24:49])
            sta_precdf[sta,3]=np.sum(rsfaipmf[sta][49:99])
            sta_precdf[sta,4]=np.sum(rsfaipmf[sta][99:249])
            sta_precdf[sta,5]=np.sum(rsfaipmf[sta][249:])
        except:
            print(str(sta)+'站无降水！')
        
        lonlat_sta_precdf=np.concatenate((lonlat,sta_precdf),axis=1)
        products_data_name=products_path+'precdf_'+str(foreday+1)+'day.txt'
        np.savetxt(products_data_name,lonlat_sta_precdf,fmt='%f')
