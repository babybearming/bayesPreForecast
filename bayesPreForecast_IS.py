# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:21:18 2019

@author: babybearming
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as s
import datetime

def obs_g(obsdata):    # 每天出现降水的先验概率计算
    datenums=len(obsdata[0,:,0])
    stanums=len(obsdata[0,0,:])
    obs_g=np.full([datenums,stanums],np.nan)
    obsdata[obsdata>0]=1
    
    for i in range(datenums):
        for j in range(stanums):
            obs_g[i,j]=np.nanmean(obsdata[:,i,j])
    return obs_g
        
def getobsdata(startyr,endyr):
    obspath='./Data/obsstadata/'
    yrnums=endyr-startyr+1  #先验概率计算背景年份
    obsdata1=[i for i in range(yrnums)]
    for inx,yr in enumerate(range(startyr,endyr+1)):
        filename=obspath+str(yr)+'.npy'
        obsdata1[inx]=np.load(filename)
    datenums=len(obsdata1[0])
    stanums=len(obsdata1[0][0])
    obsdata=np.full([yrnums,datenums,stanums],np.nan)
    for yr in range(yrnums):
        obsdata[yr]=obsdata1[yr]
    return obsdata

def getforecastdata(startyr,endyr,foredays):
    '''
    startyr:预报起始年份
    endyr:预报结束年份
    foredays:预报时效  1、2、3...10天
    '''
    forecastpath='./Data/forecastdata/'+str(foredays)+'/'
    yrnums=endyr-startyr+1
    filename1=forecastpath+str(startyr)+'.npy'
    foretmp=np.load(filename1)
    fshp=foretmp.shape
    forecastdata=np.full([yrnums,fshp[0],fshp[1],fshp[2]],np.nan)
    for inx,yr in enumerate(range(startyr,endyr+1)):
        filename=forecastpath+str(yr)+'.npy'
        fore1=np.load(filename)
        forecastdata[inx,:,:,:]=fore1
    forecastdata[forecastdata<0.1]=0
    return forecastdata

def getforecastdata_daily(nowdate,foredays):
    '''
    nowdate:预报起始日期
    foredays:预报时效  1、2、3...10天
    varnum: 集合成员编号，0为控制预报
    '''
    forecastpath='./Data/forecastdata_daily/'+str(foredays)+'/'
    pstr=nowdate.strftime('%Y%m%d')
    filename=forecastpath+pstr+'.npy'
    forecastdata11=np.load(filename)
    forecastdata11[forecastdata11<0.1]=0
    return forecastdata11
    
def g(startyr1,endyr1,sta):
    obsdata=getobsdata(startyr1,endyr1)
    gg=obs_g(obsdata)
    g1=gg[:,sta]
    return g1

#def pai(g1,obs,fore):

def Qfu(p):
    a0=2.30753
    a1=0.27061
    b1=0.99229
    b2=0.04481
    if p>0 and p<=0.5:
        t=math.sqrt(-2*(math.log(p)))
        zp=(a0+a1*t)/(1+b1*t+b2*math.pow(t,2))-t
    if p>0.5 and p<1:
        t=math.sqrt(-2*(math.log(1-p)))
        zp=t-(a0+a1*t)/(1+b1*t+b2*math.pow(t,2))
    return zp

def Q(x1):
    a=0.33267
    a1=0.436183
    a2=-0.1201676
    a3=0.9372986
    t=1/(1+a*abs(x1))
    if x1>0:
        qx=1-((a1+a2*math.pow(t,2)+a3*math.pow(t,3))/math.sqrt(2*math.pi))\
        *math.exp(-1*math.pow(x1,2)/2)
    if x1<=0:
        qx=((a1+a2*math.pow(t,2)+a3*math.pow(t,3))/math.sqrt(2*math.pi))\
        *math.exp(-1*math.pow(x1,2)/2)
    return qx

def weibull(data):
    shape,loc,scale=s.weibull_min.fit(data,floc=0)
    wei=s.weibull_min(shape,loc,scale)
#    x=np.linspace(np.min(obs),np.max(obs))
#    plt.hist(obs,normed=True,fc="none",ec="grey",label="frequency")
#    plt.plot(x,wei.cdf(x),label="cdf")
#    plt.plot(x,wei.pdf(x),label="pdf")
    return wei

def weibullplot():
    weiobs=weibull(obs)
    weifore=weibull(fore)
    xobs=np.linspace(np.min(obs),np.max(obs))
    xfore=np.linspace(np.min(fore),np.max(fore))
    plt.plot(xobs,weiobs.cdf(xobs),label="cdf")       
    plt.plot(xfore,weifore.cdf(xfore),label="cdf")     
    plt.show()

def QfuGy(data):
    wei1=weibull(data)
    yobs1=data
    
    Qfuy1=np.zeros([len(yobs1)])
    yy1=np.zeros([len(yobs1)])
    for i,y in enumerate(yobs1):
        if y==0:
            yy1[i]=wei1.pdf(y)
        else:
            yy1[i]=wei1.cdf(y)
        Qfuy1[i]=Qfu(yy1[i])
    return Qfuy1,yy1

def QfuGy_midu(data,maxpre1):
    wei1=weibull(data)
    yobs1=np.arange(1,maxpre1,1.0)
    yobs1[1:]=yobs1[:-1]
    yobs1[0]=0.01
    Qfuy1=np.zeros([len(yobs1)])
    yy1=np.zeros([len(yobs1)])
    for i,y in enumerate(yobs1):
        yy1[i]=wei1.cdf(y)
        Qfuy1[i]=Qfu(yy1[i])
    gypmf=pypmf(yy1)
    return Qfuy1,gypmf,yobs1,yy1

def QfuKx(foredata1,x1):
    wei1=weibull(foredata1)
    cdfx1=wei1.cdf(x1)
    Qfux=Qfu(cdfx1)
    return Qfux

def uzpara(obs1,fore1):
    u,yy=QfuGy(obs1)  #第二参数为最大降水概率显示范围，第三参数为降水量显示间隔float型
    z,xx=QfuGy(fore1)
    miu0=np.mean(u)
    miu1=np.mean(z)
    d0=np.var(u)
    d1=np.var(z)
    d10=np.cov(u,z)[0,1]
    dfang=d1-d10**2/d0
#    d=math.sqrt(dfang)
    a=d10/d0
    afang=a**2
    b=miu1-d10/d0*miu0
    T=math.sqrt(dfang/(afang+dfang))
    c1=a/(afang+dfang)
    c0=-1*a*b/(afang+dfang)   
    return T,c1,c0,afang,dfang



def Faiy(udata,foredata1,x,T,c1,c0):
    ulen=len(udata)
    Faiyvalue=np.zeros([ulen])
    tmpx=QfuKx(foredata1,x)
    for i in range(ulen):
        Faiyvalue[i]=Q((udata[i]-c1*tmpx-c0)/T)
    return Faiyvalue

def calFaiy(obs1,fore1,x1):
    '''
    x: 预报因子为x的条件下，计算得到Faiy，预报量的累积概率分布系列值
    x为数值型，由模式预报给定
    '''
#    weibullplot()
#    obs1=obs1[obs1>0]
#    fore1=fore1[fore1>0]
#    print('历史样本最大降水量为 '+str(maxfore)+'，请注意使用该概率预报产品!')
#    maxpre=int(np.max([maxobs,maxfore]))+10
    u,yy=QfuGy(obs1)  #第二参数为最大降水概率显示范围，第三参数为降水量显示间隔float型
    z,xx=QfuGy(fore1)
    
#    plt.plot(u,yy)
#    plt.plot(z,xx)
#    plt.show()
    
    T,c1,c0,afang,dfang=uzpara(u,z)
#    T,c1,c0,afang,dfang=uzpara(obs1,fore1)
    
    #累积概率密度生成
    obsmax=np.max(obs1)+50
    u,gypmf,y0,gycdf=QfuGy_midu(obs1,obsmax)  #第二参数为最大降水概率显示范围，第三参数为降水量显示间隔float型
    Faiydata=Faiy(u,fore1,x1,T,c1,c0)
    return y0,gypmf,gycdf,Faiydata,afang,dfang

def mergedata_obs_fore(obsdata1,forevarnum1):
    newfore=forevarnum1
    foreshp=newfore.shape
    newobs=np.full_like(newfore,np.nan)
    newobs[:,:]=obsdata1[:,0:foreshp[1]]
    newobs=newobs.reshape(-1,1)
    newfore=newfore.reshape(-1,1)
    newobs[np.isnan(newfore)]=np.nan
    newfore[np.isnan(newobs)]=np.nan
    return newobs,newfore

def datapredeal(sta,obsdata,forevarnum):
    '''
    sta: 站点编号0，1，.varnum..450  stanums=451
    foredays: 预报时效
    varnum: 集合成员编号
    '''
    obsdata1=obsdata[:,:,sta]
    forevarnum1=forevarnum[:,:,sta]
    
    newobs,newfore=mergedata_obs_fore(obsdata1,forevarnum1)
    obs=newobs
    fore=newfore
    obs=obs[~np.isnan(obs)]
    #obs=obs[obs>0]
    fore=fore[~np.isnan(fore)]
    return obs,fore
    #fore=fore[fore>0] 
    
#def getdatasnew(sta,foredays,varnum):
#    '''
#    sta: 站点编号0，1，...450  stanums=451
#    foredays: 预报时效
#    '''
#    startyr=2017
#    endyr=2018
#    numyrs=endyr-startyr+1
#    foredata1=getforecastdata(startyr,endyr,foredays)  #第三个输入变量为预报时效，1表示预报20时起报未来1天20-20时累积降水量
#    datenums=len(foredata1[0][0])
#    stanums=len(foredata1[0][0][0])
#    foredata=np.full([numyrs,datenums,stanums],np.nan)
#    for yr in range(numyrs):
#        foredata[yr,:,:]=foredata1[yr][varnum]
#    obsdata=getobsdata(startyr,endyr)
#    obsnew=getobsdata(1981,2018)
#    gg=g(1981,2018)
#    obs=obsdata[:,:,sta]
#    fore=foredata[:,:,sta]
#    g1=gg[:,sta]
#    obs=obs[~np.isnan(obs)]
#    #obs=obs[obs>0]
#    fore=fore[~np.isnan(fore)]
#    return obs,fore,g1,obsnew

def forevx(obs,fore):
    fore0=fore[obs==0]
    fore00=fore0[fore0==0]  #实况无降水，预报无降水
    fore01=fore0[fore0>0]   #实况无降水条件下，预报有降水的样本
    fore1=fore[obs>0] 
    fore10=fore1[fore1==0]  #实况有降水，预报无降水
    fore11=fore1[fore1>0]  #实况有降水条件下，预报有降水的样本
    return fore00,fore01,fore10,fore11

def Pai(g1,obs,fore,x1,datesnum):
    '''
    g1: 历史上每天的降水概率
    obs: 历史上的观测值
    fore: 预报值
    x1: 数值型，由模式预报量给定
    datesnum: 6月1日起，6月1日为0
    '''
    if x1==0:
        paivalue=0
    if x1>0:
        fore00,fore01,fore10,fore11=forevx(obs,fore)
        r1=len(fore10)/len(fore)               #实况有降水，预报因子无降水的概率
        r0=len(fore00)/len(fore)               #实况无降水，预报无降水的概率
        wei01=weibull(fore01)
        wei11=weibull(fore11)
        h0=wei01.cdf(x1)
        h1=wei11.cdf(x1)
        lx=((1-r0)/(1-r1))*(h0/h1)
        paivalue=1/(1+(1-g1[datesnum])*lx/g1[datesnum])
    return paivalue
   
def Pyconditionx(paivalue,Faiydata):
    pyvalue=(1-paivalue)+paivalue*Faiydata
    return pyvalue

def faimidu(Faiydata1):
    faimidu1=np.zeros([len(Faiydata1)])
    faimidu1[1:]=Faiydata1[:-1]
    faimiduvalue=Faiydata1-faimidu1
    return faimiduvalue

def pypmf(pyvalue1):
    pypmf1=np.zeros([len(pyvalue1)])
    pypmf1[1:]=pyvalue1[:-1]
    pypmf=pyvalue1-pypmf1
    return pypmf
#
#def main():
#    if x==0:
#        print('降水概率为0!')
#    else:
#        obs,fore=datapredeal(obsdata,forevarnum) #0.001 第一个参数为站点编号，第二个参数为预报时效，第三个为集合成员编号
#        
#        y,Faiydata,a,d=calFaiy(obs,fore,x) #0.077
#    #    plt.plot(y,Faiydata)
#    #    plt.show()
#        Is=1/math.sqrt((1/(a/d)**2+1))    
#        print(Is)
#        
#        paivalue=Pai(g1,obs,fore,x,datenum)  #0.022
#        pyvalue=Pyconditionx(paivalue,Faiydata)
#    #    plt.plot(y,pyvalue)
#    #    plt.show()
#        
#        faipmf=faimidu(Faiydata)
##                plt.plot(y,faipmf)
##                plt.ylabel('probability of pre')
##                plt.show()
##                  
#        pypmf1=pypmf(pyvalue)
##                plt.plot(y,pypmf1)
##                plt.ylabel('pre or not')
##                plt.show()
##                print('历史样本最大降水量为 '+str(maxfore)+'毫米，请合理使用该概率预报产品!')
    
nowdate=datetime.datetime(2018,6,2)  

startdate=datetime.datetime(2018,6,1)

stanums=255;varnums=51;foredaynums=10;

startyr=2017
endyr=2018

obsdata=getobsdata(startyr,endyr)
      
#sta=100
#foreday=1
#varnum=0
#g1=g(1981,2010,sta)
#foredays=foreday+1
#datenum=(nowdate-startdate).days+foredays #6月1日为0，和预报的日期一致
#foredata_daily=getforecastdata_daily(nowdate,foredays)
#foredata=getforecastdata(startyr,endyr,foredays)
#
#x=foredata_daily[varnum][sta]  #预报量，由模式预报给定
##            x=50
#forevarnum=foredata[:,varnum,:,:]
#main()

Is=np.full([stanums,foredaynums,varnums],np.nan)
rs=np.full([stanums,foredaynums,varnums],np.nan)

#sta=254  #143石家庄站,254天津

for sta in range(stanums):
    print('stationNum'+str(sta))
    g1=g(1981,2010,sta)
    for foreday in range(foredaynums):    
        print(foreday)
        foredays=foreday+1
        datenum=(nowdate-startdate).days+foredays #6月1日为0，和预报的日期一致
        foredata_daily=getforecastdata_daily(nowdate,foredays)
        
        foredata=getforecastdata(startyr,endyr,foredays)
        for varnum in range(varnums):
    #        x=foredata_daily[varnum][sta]  #预报量，由模式预报给定
            x=10
            forevarnum=foredata[:,varnum,:,:]
            if x==0:
                print('降水概率为0!')
            else:
                obs,fore=datapredeal(sta,obsdata,forevarnum) #0.001 第一个参数为站点编号，第二个参数为预报时效，第三个为集合成员编号
                T,c1,c0,afang,dfang=uzpara(obs,fore)
                
                Isvalue=1/math.sqrt(dfang/afang+1) 
    
                Is[sta,foreday,varnum]=Isvalue
        Is3min=np.nanmin(Is[sta,foreday,:]**3)
        Is3sum=np.nansum(Is[sta,foreday,:]**3)
        for varnum in range(varnums):
            rs[sta,foreday,varnum]=(Is[sta,foreday,varnum]**3-Is3min)/\
            (Is3sum-varnums*Is3min)

np.save('rs.npy',rs)

            






