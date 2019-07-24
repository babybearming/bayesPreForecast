# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd
import struct

def rangedate(startday,endday):   
    p=pd.date_range(startday,endday,freq='D')
    pstr=p.strftime('%Y%m%d')
    return pstr

def ecpre_read(filepath):
    
    varnum=51;tnum=41;ynum=111;xnum=133;haihe_ynum=25;haihe_xnum=25;
    
    filename=filepath+'ecmf_medium_surface_total_precipitation.dat'
    print(filename+' dealing...')
    pre_float=np.zeros((varnum,tnum,ynum,xnum))
    
    f=open(filename,"rb")
    for var in range(varnum):
        for t in range(tnum):
            for y in range(ynum):
                for x in range(xnum):
                    data=f.read(4)
                    pre_float[var][t][y][x]=struct.unpack("f",data)[0]
    f.close()
    
    # 各成员提取特定时效结果
    forecastdays=10
    var_pre=pre_float
    var_daypre=np.zeros((forecastdays,varnum,ynum,xnum))
    var_daypre_haihe=np.zeros((forecastdays,varnum,haihe_ynum,haihe_xnum))
    for i in range(forecastdays):
        var_daypre[i]=var_pre[:,(i+1)*4,:,:]-var_pre[:,i*4,:,:]
    var_daypre_haihe=var_daypre[:,:,66:91,80:105]
    return var_daypre_haihe

startday='2018-06-01'
endday='2018-09-30'
pstr=rangedate(startday,endday)
ecpre_tmp_path='./Data/ecpre_tmp/'
for daystr in pstr:
    ecpre_filepath='G:/grads/'+daystr+'12/'
    try:
        var_daypre=ecpre_read(ecpre_filepath)
        np.save(ecpre_tmp_path+daystr+'.npy',var_daypre)
    except:
        print(ecpre_filepath+' is empty!')
        continue