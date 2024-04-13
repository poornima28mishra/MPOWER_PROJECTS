#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


df= pd.read_csv("D:/PROJECT/PS_20174392719_1491204439457_log.csv.zip")


# In[5]:


df.columns


# In[6]:


df.info()


# In[8]:


df['step'].unique()


# In[49]:


df.isnull().sum


# In[14]:


df['type'].unique()


# In[16]:


type=df['type'].value_counts()


# In[18]:


transaction=type.index


# In[20]:


quantity=type.values


# In[21]:


import plotly.express as px


# In[28]:


px.pie(df,values=quantity,names=transaction,hole=0.5,title='DISTRIBUTION OF TRANSACTION TYPE')


# In[29]:


df=df.dropna()


# In[30]:


df


# In[34]:


df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],value=[2,4,1,5,3],inplace=True)


# In[33]:


type


# In[35]:


df


# In[38]:


df['isFraud']=df['isFraud'].map({0:'no fraud', 1:'fraud'})


# In[39]:


df


# In[40]:


x=df[['type','amount','oldbalanceOrg','newbalanceOrig']]


# In[41]:


y=df.iloc[:,-2]


# In[42]:


y


# In[43]:


from sklearn.tree import DecisionTreeClassifier


# In[44]:


model=DecisionTreeClassifier()


# In[45]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


# In[46]:


model.fit(xtrain,ytrain)


# In[47]:


model.score(xtest,ytest)


# In[48]:


model.predict([[2,9800,170136,160296]])

