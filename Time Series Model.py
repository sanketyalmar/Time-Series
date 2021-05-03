#!/usr/bin/env python
# coding: utf-8

# # Assignment For Data Science Internship

# In[214]:


#Imorting Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[215]:


pwd


# In[216]:


data=pd.read_csv("delivery_data.csv")


# In[217]:


data.head()


# # Data Exploration

# In[218]:


#checking null values present or not.
data.isnull().sum()


# In[219]:


data.describe()


# In[220]:


#Univariate Analysis
data['TEMPERATURE'].plot.hist()


# In[221]:


#Univariate Analysis
data['WIND_SPEED'].plot.hist()


# In[222]:


#Univariate Analysis
data['PRECIPITATION'].plot.hist()


# In[223]:


#Univariate Analysis
data['ITEM_COUNT_DELIVERED'].plot.hist()


# In[224]:


#Correlation
data.corr()


# In[225]:


#Bivariate Analysis
data.plot(x ='TIMESTAMP', y='ITEM_COUNT_DELIVERED', kind = 'line')


# In[226]:


#Bivariate Analysis
data.plot(x ='PRECIPITATION', y='ACTUAL_DELIVERY_MINUTES', kind = 'scatter')


# In[227]:


#Bivariate Analysis
data.plot(x ='TEMPERATURE', y='ACTUAL_DELIVERY_MINUTES', kind = 'scatter')


# In[228]:


#Bivariate Analysis
data.plot(x ='WIND_SPEED', y='ACTUAL_DELIVERY_MINUTES', kind = 'scatter')


# In[229]:


#Bivariate Analysis
data.plot(x ='USER_LAT', y='ITEM_COUNT_DELIVERED', kind = 'scatter')


# In[230]:


#Bivariate Analysis
data.plot(x ='PRECIPITATION', y='ITEM_COUNT_DELIVERED', kind = 'scatter')


# In[231]:


#Bivariate Analysis
data.plot(x ='TEMPERATURE', y='ITEM_COUNT_DELIVERED', kind = 'scatter')


# In[232]:


#Bivariate Analysis
data.plot(x ='WIND_SPEED', y='ITEM_COUNT_DELIVERED', kind = 'scatter')


# # Seasonal Decompose

# In[233]:


#Decomposition is nothing but additive and multiplicative parts of time series. 
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


# In[234]:


item_count_delivered = pd.read_csv('delivery_data.csv', parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')


# In[235]:


item_count_delivered.head(10)


# In[236]:


item_count_delivered.reset_index(inplace=True)


# In[237]:


import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize': (10,6)})
plt.plot(item_count_delivered['ITEM_COUNT_DELIVERED'])


# In[238]:


# Multiplicative Decomposition 
mul_result = seasonal_decompose(item_count_delivered['ITEM_COUNT_DELIVERED'], model='multiplicative',period=1)

# Additive Decomposition
add_result = seasonal_decompose(item_count_delivered['ITEM_COUNT_DELIVERED'], model='additive',period=1)


# In[239]:


plt.rcParams.update({'figure.figsize': (10,10)})
mul_result.plot().suptitle('\nMultiplicative Decompose', fontsize=12)


# In[240]:


add_result.plot().suptitle('\nAdditive Decompose', fontsize=12)
plt.show()


# In[241]:


new_df_add = pd.concat([add_result.seasonal, add_result.trend, add_result.resid, add_result.observed], axis=1)
new_df_add.columns = ['seasoanilty', 'trend', 'residual', 'actual_values']
new_df_add.head(5)


# In[242]:


new_df_mult = pd.concat([mul_result.seasonal, mul_result.trend, mul_result.resid, mul_result.observed], axis=1)
new_df_mult.columns = ['seasoanilty', 'trend', 'residual', 'actual_values']
new_df_mult.head(5)


# #From seasonal decompose (both i.e. additive and multiplicative) we can  say that time series is more or less simmilar to trend component,which means that trend component is more prominent and seasonal component is less prominent.

# # Stationary Check
#To build algorithm for time series the data should be stationary and to have data stationary it should have constant mean and constant variance.Here we are doing stationary check using Adfuller method:
# In[243]:


from statsmodels.tsa.stattools import adfuller


# In[244]:


test_result=adfuller(data['ITEM_COUNT_DELIVERED'])


# In[245]:


# ADF Test - nul hypothesis - non-stationary - if p-value < 5% reject null hypothesis
def adfuller_test(ITEM_COUNT_DELIVERED):
    result=adfuller(ITEM_COUNT_DELIVERED)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[246]:


adfuller_test(data['ITEM_COUNT_DELIVERED'])


# #So from stationary check by using adfuller method, we come to know that given data is statinary so, now we can move to modelling part. 

# # Modelling

# #There are various methods for time series modelling like:
# 1.Autoregression (AR)
# 2.Moving Average (MA)
# 3.Autoregressive Moving Average (ARMA)
# 4.Autoregressive Integrated Moving Average (ARIMA)
# 5.Seasonal Autoregressive Integrated Moving-Average (SARIMA)
# 6.Vector Autoregression (VAR)
# 7.Vector Autoregression Moving-Average (VARMA)
# 8.Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)
# 9.Simple Exponential Smoothing (SES)
# 10.Holt Winter’s Exponential Smoothing (HWES)
# Out of these all these methods only SARIMA and HWES methods considers seasonal and trend component that's 
# why we are using these method for modelling purpose. 

# In[247]:


# Modelling by SARIMA method,
import statsmodels.api as sm
from random import random


# In[248]:


#1.How many orders are we expected to get in September 2020 - first week? 
model=sm.tsa.statespace.SARIMAX(data['ITEM_COUNT_DELIVERED'],order=(1, 1, 1),seasonal_order=(1,1,1,3))
results=model.fit(disp=False)
predictions = results.predict(start = 0, end = 6)
predictions


# In[249]:


#Ans:
sum(predictions)


# 2.What are the expected peak times where the restaurant should gather more resources for deliveries in September first week?
# #Ans:From the above predictions we can say that on 5th day of week restaurant should gather more resources for deliveries in September first week.

# In[250]:


data['ITEM_COUNT_DELIVERED'].max()


# In[251]:


data['ACTUAL_DELIVERY_MINUTES'].max()


# 3.Out of the expected orders what percentage and which orders will be delivered within an hour?
# Ans:from given data and from the above two lines of codes we can say that max 58min is taken for 11 deliveres, so from the expected orders all the orders will be delivered within an hour.

# # Further Development

# In[252]:


#Here we are using HWES method to check whethere any improvement in performance can be possible or not.


# In[253]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
from random import random


# In[254]:


data = [x + random() for x in range(1, 100)]
model = ExponentialSmoothing(data)


# In[255]:


model_fit = model.fit()


# In[256]:


predictions2 = model_fit.predict(start = 0, end = 6)
predictions2


# In[257]:


sum(predictions2)


# # Evaluation

# #The expected result based on intuition should be in the range between 35 to 77.This observation is based on minimum and maximum per week count.Therefore the applied time series model is not suitable for estimation of provided assignment.

# In[ ]:




