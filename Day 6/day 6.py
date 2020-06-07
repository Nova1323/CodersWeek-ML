#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle


# In[2]:


ds = pd.read_csv('http://iali.in/datasets/Social_Network_Ads.csv')
ds = ds[['Gender','Age','EstimatedSalary','Purchased']]


# In[3]:


gender = {'Male': 1,'Female': 0}
ds.Gender = [gender[item] for item in ds.Gender] 


# In[4]:


X = ds.iloc[:,0:3].values
Y = ds.iloc[:,-1].values
ds.head()


# In[5]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1)


# In[6]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
dtpredictions = dt.predict(X_test)


# In[7]:


accuracy_score(dtpredictions,Y_test)


# In[8]:


pickle.dump(dt,open('model.pkl','wb'))


# In[9]:


model=pickle.load(open('model.pkl','rb'))


# In[ ]:




