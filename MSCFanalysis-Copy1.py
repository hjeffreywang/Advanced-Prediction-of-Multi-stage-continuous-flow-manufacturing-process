#!/usr/bin/env python
# coding: utf-8

# In[1]:



from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


# continuous_factory_process.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('data/continuous_factory_process.csv', delimiter=',')
df1.dataframeName = 'continuous_factory_process.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[3]:


df1.head(5)


# In[4]:


# deleting the first column

df1 = df1.drop(columns = ['time_stamp'], axis = 1)

# checking the shape of the data after deleting a column
df1.shape


# In[5]:


#filter the y variables (Output measurement actuals) for prediction

df2=df1.filter(regex='Stage1', axis=1)
df2


# In[6]:


#create list of columns names for x and y separation
ylist=df2.columns.to_list()
range(len(ylist)-1)


# In[7]:


# separating the dependent and independent data
x=df1.drop(ylist, axis=1)
y = df2


# In[8]:

df3=df2.iloc[:,1:2]

df3

