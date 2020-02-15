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

df2=df1.filter(regex='Stage1', axis=1).filter(regex='Actual', axis=1)
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


x


# In[9]:


y.iloc[:,14]


# In[10]:


# splitting them into train test and split each 15 

from sklearn.model_selection import train_test_split


    
x_train, x_test, y_train, y_test = train_test_split(x, y.iloc[:,1], test_size = 0.2, random_state = 0)

# gettiing the shapes
print("shape of x_train: ", x_train.shape)
print("shape of x_test: ", x_test.shape)
print("shape of y_train: ", y_train.shape)
print("shape of y_test: ", y_test.shape)


# In[11]:


x_train.head


# In[12]:


y_train


# In[13]:


# standardization of 15 variables

from sklearn.preprocessing import StandardScaler

# creating a standard scaler
sc = StandardScaler()

# fitting independent data to the model
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[27]:


y_train.shape


import xgboost as xgb
from xgboost.sklearn import XGBClassifier

model = XGBClassifier()

model.fit(x_train[0:7000], y_train.iloc[0:7000])

y_pred = model.predict(x_test)
y_pred.shape



x_train[0:1000]


y_train[0:1000]



from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)





# In[ ]:


from sklearn.metrics import mean_squared_error

score = mean_squared_error(y_test, y_pred)
np.sqrt(score)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:




