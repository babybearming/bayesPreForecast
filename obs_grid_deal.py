# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 20:01:04 2019
select region grid data form grid files
@author: babybearming
"""

import numpy as np 
import pandas as pd

def rangedate(startday,endday):   
    p=pd.date_range(startday,endday,freq='D')
    pstr=p.strftime('%Y%m%d')
    return pstr

for yr in range(2018,2019):
    year=yr
    yearstr=str(year)
    
    ynum=17;xnum=17;
    obs_grid_path='./Data/obs_grid/'+yearstr+'/' #格点文件路径
    filepre='SURF_CLI_CHN_PRE_DAY_GRID_0.5-' #文件前缀
    
    startday=yearstr+'-06-01'  #遍历文件起始日期
    endday=yearstr+'-10-10'   #结束日期
    pstr=rangedate(startday,endday)
    tnum=len(pstr)
    
    obsdata=np.full([tnum,ynum,xnum],np.nan)
    for inx,daystr in enumerate(pstr):
        filename=obs_grid_path+filepre+daystr+'.txt'
        try:
            txtdata=np.loadtxt(filename,skiprows=6,dtype=float)
            obsdata[inx]=txtdata[34:51,80:97] #海河流域范围格点
        except:
            print(filename+' is not exsit!')
            continue
    
    np.save('./Data/obsgriddata/'+yearstr+'.npy',obsdata)