# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:18:26 2019

@author: babybearming
"""
import numpy as np 
import pandas as pd
from metpy.interpolate import (interpolate_to_points)
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

def rangedate(startday,endday):   
    p=pd.date_range(startday,endday,freq='D')
    pstr=p.strftime('%Y%m%d')
    return pstr

datapath,folder_name,ele_name,stationid_name=iniset()
stationid=stationidread(datapath,stationid_name)
xi=stationid[['lon','lat']].values
stanums=len(xi)

year=2018
yearstr=str(year)

startday=yearstr+'-06-01'
endday=yearstr+'-09-30'

ynum=25;xnum=25;tforenum=10;varnum=51;
x_grid=np.arange(110,122.5,0.5)
y_grid=np.arange(33,45.5,0.5)

lon,lat=np.meshgrid(x_grid,y_grid)
lon=lon.reshape((-1,1))
lat=lat.reshape((-1,1))
lonlat=np.concatenate([lon,lat],axis=1)

ecpre_tmp_path='./Data/ecpre_tmp/'

pstr=rangedate(startday,endday)
tnum=len(pstr)

stadatacell=[i for i in range(tforenum)]
for tt in range(tforenum):
    print(tt)
    stadata=np.full([varnum,tnum,stanums],np.nan)
    for var in range(varnum):
        for inx,daystr in enumerate(pstr):
            filename=ecpre_tmp_path+daystr+'.npy'
            try:
                M=np.load(filename)
                fredata=M[tt,var].reshape((-1,1))
                stadata[var,inx,:]=list(interpolate_to_points(lonlat,fredata,xi))
            except:
#                print(filename+' is not exsit!')
                continue
    stadatacell[tt]=stadata
#
stafore2obs=[i for i in range(tforenum)]
for tt in range(tforenum):
    stafore2obs[tt]=np.full([varnum,tnum+tt+1,stanums],np.nan)
    foretmp=np.full([varnum,tnum+tt+1,stanums],np.nan)
    foretmp[:,tt+1:,:]=stadatacell[tt]
    stafore2obs[tt]=foretmp

for tt in range(tforenum):
    filepath='./Data/forecastdata/'+str(tt+1)+'/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    np.save(filepath+yearstr+'.npy',stafore2obs[tt])
