# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 20:22:46 2019

@author: babybearming
"""
import numpy as np

# 在运行bayesPreForecast后使用
# 后验概率生成
fore2018=foredata[1][0][:,254]  #143石家庄，254天津


houyangl=np.full([153,1],np.nan)

for inx,x in enumerate(fore2018):
    try:
        houyangl[inx]=Pai(g1,obs,fore,x,inx)
    except:
        print('nan')
        
#散点图
        
plt.scatter(obs,fore,color='',edgecolor='k')
plt.xlabel('Precip.Amount w [mm]')
plt.ylabel('24-h Total Precip.x [mm]')
plt.show()


u,yy,y0=QfuGy(obs)  #第二参数为最大降水概率显示范围，第三参数为降水量显示间隔float型
z,xx,x0=QfuGy(fore)
plt.scatter(u,z,color='',edgecolor='k')
plt.xlabel('u=$Q^{-1}$(G(w))')
plt.ylabel('z=$Q^{-1}$(K(x))')


T,c1,c0,afang,dfang=uzpara(u,z)