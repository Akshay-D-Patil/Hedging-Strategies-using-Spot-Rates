#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Tue Apr 10 02:34:54 2018



                                                        FINN 6112 - Spring 2018



                                                        Topic: Hedging strategies

                                          

                                                       Author: Akshay Patil

                                                      Instructor: Dr.Ethan Chiang



"""



import pandas as pd

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.formula.api as sm

import os

os.chdir('/Users/khan/Desktop/Sem-II/FINN 6212')



# Importing data



data=pd.read_excel("/Users/khan/Desktop/Sem-II/FINN 6212/feds200628.xls",index_col=None,header=9)



data.reset_index(inplace = True)



data.columns.values[0] = 'date'



#Sorting data with respect to date 



data.date = pd.to_datetime(data.date)



data['year'] = data.date.dt.year

data['month'] = data.date.dt.month

data['day'] = data.date.dt.day



# Keep month end dates only



data['rank'] = data.groupby(['year','month'])['day'].rank(ascending=0,method='dense')



data = data[data['rank']==1]



# Keep only required columns



data=data[['date','BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']]



# Creating the spot rates and price of the bond for the given maturities



data = data.sort_values('date')



for i in (1/12,1/4,11/12,1,2,35/12,3,59/12,5,83/12,7,8,119/12,10):

     x1 = i / data['TAU1']

     x2 = i / data['TAU2']

     

     data['y(t)-'+str(i)] = data['BETA0'] + data['BETA1'] * ((1 - (np.exp(-x1))) /  x1 ) + data['BETA2'] * (((1 - (np.exp(-x1))) / x1) - (np.exp(-x1))) + data['BETA3'] * ((((1 - (np.exp(-x2)))) / x2 )  - (np.exp(- x2)))  

     data['P(t)-'+str(i)] = np.exp(-i * (data['y(t)-'+str(i)]))

     

# Calculating the excess return for the given maturities



data['ERET(1)']=0

data['ERET(3)']=0

data['ERET(5)']=0

data['ERET(7)']=0

data['ERET(10)']=0

    

for i in (1,3,5,7,10):

    for j in range(0,len(data)-1):

        data.iloc[j+1,data.columns.get_loc('ERET('+str(i)+')')] = (data.iloc[j+1,data.columns.get_loc('P(t)-'+str(i-(1/12)))] / data.iloc[j,data.columns.get_loc('P(t)-'+str(i))]) - (1/data.iloc[j,data.columns.get_loc('P(t)-'+str(1/12))])

    

    

#Calcualting the level, slope and curvature

    

data['level'] = data['y(t)-'+str(1/4)]

data['slope'] = data['y(t)-'+str(8)] - data['y(t)-'+str(1/4)]

data['curvature'] = (data['y(t)-'+str(8)] - data['y(t)-'+str(2)]) -  (data['y(t)-'+str(2)] - data['y(t)-'+str(1/4)])



data['level_d'] = 0

data['slope_d'] = 0

data['curvature_d'] = 0

for i in range(1,len(data)):



    data.iloc[i-1, data.columns.get_loc('level_d')] = data.iloc[i, data.columns.get_loc('level')]

    data.iloc[i-1, data.columns.get_loc('slope_d')] = data.iloc[i, data.columns.get_loc('slope')]

    data.iloc[i-1, data.columns.get_loc('curvature_d')] = data.iloc[i, data.columns.get_loc('curvature')]



# auto correlation for the spot rates



for j in (1,5,10):

    for i in range(1,5):

        autocorr = data['y(t)-'+str(j)].autocorr(lag=i)

        print('The autocorrelation of spot rate y(t)-%d at lag %d is %f' % (j,i,autocorr))



print(data[['y(t)-1','y(t)-5','y(t)-10']].corr())

   

#Modified and Macaulay duration and weights for barbell strategy 

    

for i in (1,5,10):



    data['mod_dur_y(t)-'+str(i)] = i / (1 + (data['y(t)-'+str(i)] / 100 ))

    data['mac_dur_y(t)-'+str(i)] = i

    

    #descriptive statistics of spot rates

    mean = data['y(t)-'+str(i)].mean()

    std_dev = data['y(t)-'+str(i)].std()

    skewness = data['y(t)-'+str(i)].skew()

    kurtosis = data['y(t)-'+str(i)].kurtosis()

    print('The mean of spot rate y(t)-%d is %f' % (i,mean))

    print('The standard deviation of spot rate y(t)-%d is %f' % (i,std_dev))

    print('The skewness of spot rate y(t)-%d is %f' % (i,skewness))

    print('The kurtosis of spot rate y(t)-%d is %f\n' % (i,kurtosis))

    

y1 = data['y(t)-1']

y5 = data['y(t)-5']

y10 = data['y(t)-10']

level = data['level']

slope = data['slope']

curvature = data['curvature']





#Model for y(t)-1 



model_y1 = sm.ols(formula="y1 ~ level + slope + curvature ", data=data).fit()



print(model_y1.summary())



#Model for y(t)-5 



model_y5 = sm.ols(formula="y5 ~ level + slope + curvature ", data=data).fit()



print(model_y5.summary())



#Model for y(t)-1 



model_y10 = sm.ols(formula="y10 ~ level + slope + curvature ", data=data).fit()



print(model_y10.summary())       



#weights of modifiied duration  

    

data['mod_dur_w1'] = (data['mod_dur_y(t)-5'] - data['mod_dur_y(t)-10']) / (data['mod_dur_y(t)-1'] - data['mod_dur_y(t)-10'])

data['mod_dur_w10'] = 1 - data['mod_dur_w1']





#weights of macaulay duration  



data['mac_dur_w1'] = (data['mac_dur_y(t)-5'] - data['mac_dur_y(t)-10']) / (data['mac_dur_y(t)-1'] - data['mac_dur_y(t)-10'])

data['mac_dur_w10'] = 1 - data['mac_dur_w1']   

    



#test, train = np.split(data,2)



#Model for ERET(3) and the corresponding weights for hedging strategy 1



j=0

l =len(data)



data['error(3)-h1']=0

data['RMSE(3)-h1']=0

for i in range(int(l/2),l):

    train3_h2 = data.iloc[j:i-1,:]

    data.iloc[i, data.columns.get_loc('error(3)-h1')] = data.iloc[i, data.columns.get_loc('ERET(3)')] - ((data.iloc[i-1, data.columns.get_loc('mod_dur_y(t)-1')] *  data.iloc[i, data.columns.get_loc('ERET(1)')])  + (data.iloc[i-1, data.columns.get_loc('mod_dur_y(t)-10')] * data.iloc[i, data.columns.get_loc('ERET(10)')]))

    data.iloc[i, data.columns.get_loc('RMSE(3)-h1')] = data.iloc[i, data.columns.get_loc('error(3)-h1')] * data.iloc[i, data.columns.get_loc('error(3)-h1')]

    j=j+1



x1 = np.sqrt(sum(data['RMSE(3)-h1'])/(len(data)/2))

print('\n The RMSE(3) of h1 is %f' % x1)

     

#Model for ERET(7) and the corresponding weights for hedging strategy 1



j=0

l =len(data)



data['error(7)-h1']=0

data['RMSE(7)-h1']=0

for i in range(int(l/2),l):

    train3_h2 = data.iloc[j:i-1,:]

    data.iloc[i, data.columns.get_loc('error(7)-h1')] = data.iloc[i, data.columns.get_loc('ERET(7)')] - ((data.iloc[i-1, data.columns.get_loc('mod_dur_y(t)-1')] *  data.iloc[i, data.columns.get_loc('ERET(1)')])  + (data.iloc[i-1, data.columns.get_loc('mod_dur_y(t)-10')] * data.iloc[i, data.columns.get_loc('ERET(10)')]))

    data.iloc[i, data.columns.get_loc('RMSE(7)-h1')] = data.iloc[i, data.columns.get_loc('error(7)-h1')] * data.iloc[i, data.columns.get_loc('error(7)-h1')]

    j=j+1

    

x2 = np.sqrt(sum(data['RMSE(7)-h1'])/(len(data)/2))

print('\n The RMSE(7) of h1 is %f' % x2)



#Model for ERET(3) and the corresponding weights for hedging strategy 2



j=0

l =len(data)



data['error(3)-h2']=0

data['RMSE(3)-h2']=0

for i in range(int(l/2),l):

    train3_h2 = data.iloc[j:i-1,:]

    data.iloc[i, data.columns.get_loc('error(3)-h2')] = data.iloc[i, data.columns.get_loc('ERET(3)')] - ((data.iloc[i-1, data.columns.get_loc('mac_dur_y(t)-1')] *  data.iloc[i, data.columns.get_loc('ERET(1)')])  + (data.iloc[i-1, data.columns.get_loc('mac_dur_y(t)-10')] * data.iloc[i, data.columns.get_loc('ERET(10)')]))

    data.iloc[i, data.columns.get_loc('RMSE(3)-h2')] = data.iloc[i, data.columns.get_loc('error(3)-h2')] * data.iloc[i, data.columns.get_loc('error(3)-h2')]

    j=j+1

    

x3 = np.sqrt(sum(data['RMSE(3)-h2'])/(len(data)/2))

print('\n The RMSE(3) of h2 is %f' % x3)

     

#Model for ERET(7) and the corresponding weights for hedging strategy 2



j=0

l =len(data)



data['error(7)-h2']=0

data['RMSE(7)-h2']=0

for i in range(int(l/2),l):

    train3_h2 = data.iloc[j:i-1,:]

    data.iloc[i, data.columns.get_loc('error(7)-h2')] = data.iloc[i, data.columns.get_loc('ERET(7)')] - ((data.iloc[i-1, data.columns.get_loc('mac_dur_y(t)-1')] *  data.iloc[i, data.columns.get_loc('ERET(1)')])  + (data.iloc[i-1, data.columns.get_loc('mac_dur_y(t)-10')] * data.iloc[i, data.columns.get_loc('ERET(10)')]))

    data.iloc[i, data.columns.get_loc('RMSE(7)-h2')] = data.iloc[i, data.columns.get_loc('error(7)-h2')] * data.iloc[i, data.columns.get_loc('error(7)-h2')]

    j=j+1



x4 = np.sqrt(sum(data['RMSE(7)-h2'])/(len(data)/2))

print('\n The RMSE(7) of h2 is %f' % x4)



#Model for ERET(3) and the corresponding weights for hedging strategy 3



j=0

l =len(data)



data['error(3)-h3']=0

data['RMSE(3)-h3']=0

for i in range(int(l/2),l):



    train3_h3 = data.iloc[j:i-1,:]

    model3_h3 = sm.ols(formula="train3_h3['ERET(3)'] ~ train3_h3['ERET(1)'] + train3_h3['ERET(5)'] + train3_h3['ERET(10)'] - 1", data=train3_h3).fit()

    j=j+1

    data.iloc[i, data.columns.get_loc('error(3)-h3')] = data.iloc[i, data.columns.get_loc('ERET(3)')] - (( model3_h3.params[0] * data.iloc[i, data.columns.get_loc('ERET(1)')]) + ( model3_h3.params[1] * data.iloc[i, data.columns.get_loc('ERET(5)')]) + ( model3_h3.params[2] * data.iloc[i, data.columns.get_loc('ERET(10)')]))

    data.iloc[i, data.columns.get_loc('RMSE(3)-h3')] = data.iloc[i, data.columns.get_loc('error(3)-h3')] * data.iloc[i, data.columns.get_loc('error(3)-h3')]



x5 = np.sqrt(sum(data['RMSE(3)-h3'])/(len(data)/2))

print('\n The RMSE(3) of h3 is %f' % x5)



#Model for ERET(7) and the corresponding weights for hedging strategy 3



j=0

l =len(data)



data['error(7)-h3']=0

data['RMSE(7)-h3']=0

for i in range(int(l/2),l):



    train7_h3 = data.iloc[j:i-1,:]

    model7_h3 = sm.ols(formula="train7_h3['ERET(7)'] ~ train7_h3['ERET(1)'] + train7_h3['ERET(5)'] + train7_h3['ERET(10)'] - 1", data=train7_h3).fit()

    j=j+1

    data.iloc[i, data.columns.get_loc('error(7)-h3')] = data.iloc[i, data.columns.get_loc('ERET(7)')] - (( model7_h3.params[0] * data.iloc[i, data.columns.get_loc('ERET(1)')]) + ( model7_h3.params[1] * data.iloc[i, data.columns.get_loc('ERET(5)')]) + ( model7_h3.params[2] * data.iloc[i, data.columns.get_loc('ERET(10)')]))

    data.iloc[i, data.columns.get_loc('RMSE(7)-h3')] = data.iloc[i, data.columns.get_loc('error(7)-h3')] * data.iloc[i, data.columns.get_loc('error(7)-h3')]

    



x6 = np.sqrt(sum(data['RMSE(7)-h3'])/(len(data)/2))

print('\n The RMSE(7) of h3 is %f' % x6)



#Model for ERET(3) and the corresponding weights for hedging strategy 4





j=0

l =len(data)



data['error(3)-h4']=0

data['RMSE(3)-h4']=0

for i in range(int(l/2),l):

    

    train3_h4 = data.iloc[j:i-1,:]

    ER3 = train3_h4['ERET(3)'] 

    ER1_3 = train3_h4['ERET(1)'] 

    ER1_3l = train3_h4['ERET(1)'] * train3_h4['level_d']

    ER1_3s = train3_h4['ERET(1)'] * train3_h4['slope_d']

    ER1_3c = train3_h4['ERET(1)'] * train3_h4['curvature_d']

    ER5_3 = train3_h4['ERET(5)'] 

    ER5_3l = train3_h4['ERET(5)'] * train3_h4['level_d']

    ER5_3s = train3_h4['ERET(5)'] * train3_h4['slope_d']

    ER5_3c = train3_h4['ERET(5)'] * train3_h4['curvature_d']

    ER10_3 = train3_h4['ERET(10)'] 

    ER10_3l = train3_h4['ERET(10)'] * train3_h4['level_d']

    ER10_3s = train3_h4['ERET(10)'] * train3_h4['slope_d']

    ER10_3c = train3_h4['ERET(10)'] * train3_h4['curvature_d']



    model3_h4 = sm.ols(formula="ER3 ~ ER1_3 + ER1_3l + ER1_3s + ER1_3c + ER5_3 + ER5_3l + ER5_3s + ER5_3c + ER10_3 + ER10_3l + ER10_3s + ER10_3c - 1", data=train3_h4).fit()

    j=j+1

    data.iloc[i, data.columns.get_loc('error(3)-h4')] = data.iloc[i, data.columns.get_loc('ERET(3)')] - (( model3_h4.params[0] * data.iloc[i, data.columns.get_loc('ERET(1)')]) + ( model3_h4.params[1] * data.iloc[i, data.columns.get_loc('ERET(1)')] * data.iloc[i, data.columns.get_loc('level_d')] ) +  ( model3_h4.params[2] * data.iloc[i, data.columns.get_loc('ERET(1)')] * data.iloc[i, data.columns.get_loc('slope_d')] ) +  ( model3_h4.params[3] * data.iloc[i, data.columns.get_loc('ERET(1)')] * data.iloc[i, data.columns.get_loc('curvature_d')] ) +

                                                        ( model3_h4.params[4] * data.iloc[i, data.columns.get_loc('ERET(5)')]) + ( model3_h4.params[5] * data.iloc[i, data.columns.get_loc('ERET(5)')] * data.iloc[i, data.columns.get_loc('level_d')]) + ( model3_h4.params[6] * data.iloc[i, data.columns.get_loc('ERET(5)')] * data.iloc[i, data.columns.get_loc('slope_d')]) + ( model3_h4.params[7] * data.iloc[i, data.columns.get_loc('ERET(5)')] * data.iloc[i, data.columns.get_loc('curvature_d')]) +

                                                        ( model3_h4.params[8] * data.iloc[i, data.columns.get_loc('ERET(10)')]) + ( model3_h4.params[9] * data.iloc[i, data.columns.get_loc('ERET(10)')] * data.iloc[i, data.columns.get_loc('level_d')]) + ( model3_h4.params[10] * data.iloc[i, data.columns.get_loc('ERET(10)')] * data.iloc[i, data.columns.get_loc('slope_d')]) + ( model3_h4.params[11] * data.iloc[i, data.columns.get_loc('ERET(10)')] * data.iloc[i, data.columns.get_loc('curvature_d')]))

    data.iloc[i, data.columns.get_loc('RMSE(3)-h4')] = data.iloc[i, data.columns.get_loc('error(3)-h4')] * data.iloc[i, data.columns.get_loc('error(3)-h4')]



    

x7 = np.sqrt(sum(data['RMSE(3)-h4'])/(len(data)/2))

print('\n The RMSE(3) of h4 is %f' % x7)

    

#Model for ERET(7) and the corresponding weights for hedging strategy 4    

 

j=0

l =len(data)



data['error(7)-h4']=0

data['RMSE(7)-h4']=0

for i in range(int(l/2),l):

    

    train7_h4 = data.iloc[j:i-1,:]

    ER7 = train7_h4['ERET(7)'] 

    ER1_7 = train7_h4['ERET(1)'] 

    ER1_7l = train7_h4['ERET(1)'] * train7_h4['level_d']

    ER1_7s = train7_h4['ERET(1)'] * train7_h4['slope_d']

    ER1_7c = train7_h4['ERET(1)'] * train7_h4['curvature_d']

    ER5_7 = train7_h4['ERET(5)'] 

    ER5_7l = train7_h4['ERET(5)'] * train7_h4['level_d']

    ER5_7s = train7_h4['ERET(5)'] * train7_h4['slope_d']

    ER5_7c = train7_h4['ERET(5)'] * train7_h4['curvature_d']

    ER10_7 = train7_h4['ERET(10)'] 

    ER10_7l = train7_h4['ERET(10)'] * train7_h4['level_d']

    ER10_7s = train7_h4['ERET(10)'] * train7_h4['slope_d']

    ER10_7c = train7_h4['ERET(10)'] * train7_h4['curvature_d']



    model7_h4 = sm.ols(formula="ER7 ~ ER1_7 + ER1_7l + ER1_7s + ER1_7c + ER5_7 + ER5_7l + ER5_7s + ER5_7c + ER10_7 + ER10_7l + ER10_7s + ER10_7c - 1", data=train7_h4).fit()

    j=j+1

    data.iloc[i, data.columns.get_loc('error(7)-h4')] = data.iloc[i, data.columns.get_loc('ERET(3)')] - (( model7_h4.params[0] * data.iloc[i, data.columns.get_loc('ERET(1)')]) + ( model7_h4.params[1] * data.iloc[i, data.columns.get_loc('ERET(1)')] * data.iloc[i, data.columns.get_loc('level_d')] ) +  ( model7_h4.params[2] * data.iloc[i, data.columns.get_loc('ERET(1)')] * data.iloc[i, data.columns.get_loc('slope_d')] ) +  ( model7_h4.params[3] * data.iloc[i, data.columns.get_loc('ERET(1)')] * data.iloc[i, data.columns.get_loc('curvature_d')] ) +

                                                        ( model7_h4.params[4] * data.iloc[i, data.columns.get_loc('ERET(5)')]) + ( model7_h4.params[5] * data.iloc[i, data.columns.get_loc('ERET(5)')] * data.iloc[i, data.columns.get_loc('level_d')]) + ( model7_h4.params[6] * data.iloc[i, data.columns.get_loc('ERET(5)')] * data.iloc[i, data.columns.get_loc('slope_d')]) + ( model7_h4.params[7] * data.iloc[i, data.columns.get_loc('ERET(5)')] * data.iloc[i, data.columns.get_loc('curvature_d')]) + 

                                                        ( model7_h4.params[8] * data.iloc[i, data.columns.get_loc('ERET(10)')]) + ( model7_h4.params[9] * data.iloc[i, data.columns.get_loc('ERET(10)')] * data.iloc[i, data.columns.get_loc('level_d')]) + ( model7_h4.params[10] * data.iloc[i, data.columns.get_loc('ERET(10)')] * data.iloc[i, data.columns.get_loc('slope_d')]) + ( model7_h4.params[11] * data.iloc[i, data.columns.get_loc('ERET(10)')] * data.iloc[i, data.columns.get_loc('curvature_d')]))

    data.iloc[i, data.columns.get_loc('RMSE(7)-h4')] = data.iloc[i, data.columns.get_loc('error(7)-h4')] * data.iloc[i, data.columns.get_loc('error(7)-h4')]



x8 = np.sqrt(sum(data['RMSE(7)-h4'])/(len(data)/2))

print('\n The RMSE(7) of h4 is %f' % x8)

 

#Time series of plot of y(t)-1

    

plt.figure()	

plt.plot(data['date'],data['y(t)-1'])

plt.xlabel('Year')

plt.ylabel('yt(1)')

plt.title('Time series plot of yt(1)')

plt.grid(True)



#Time series of plot of y(t)-5

    

plt.figure()	

plt.plot(data['date'],data['y(t)-5'])

plt.xlabel('Year')

plt.ylabel('yt(5)')

plt.title('Time series plot of yt(5)')

plt.grid(True)



#Time series of plot of y(t)-10

    

plt.figure()	

plt.plot(data['date'],data['y(t)-10'])

plt.xlabel('Year')

plt.ylabel('yt(10)')

plt.title('Time series plot of yt(10)')

plt.grid(True)



data.to_csv('final1.csv', sep='\t')







