#!/usr/bin/env python
# coding: utf-8

# In[134]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[135]:


ds = pd.read_csv('http://iali.in/datasets/Social_Network_Ads.csv')
ds = ds[['Gender','Age','EstimatedSalary','Purchased']]


# In[136]:


gender = {'Male': 1,'Female': 0}
ds.Gender = [gender[item] for item in ds.Gender] 


# In[137]:


X = ds.iloc[:,0:3].values
Y = ds.iloc[:,-1].values
ds.head()


# In[138]:


plt.plot(X_train,'o')


# In[139]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1)


# In[140]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knnpredictions = knn.predict(X_test)


# In[141]:


accuracy_score(knnpredictions,Y_test)


# In[142]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
dtpredictions = dt.predict(X_test)


# In[143]:


accuracy_score(dtpredictions,Y_test)


# In[144]:


from sklearn.svm import SVC
svm = SVC(gamma = 'auto')
svm.fit(X_train, Y_train)
svmpredictions = svm.predict(X_test)


# In[145]:


accuracy_score(svmpredictions,Y_test)

