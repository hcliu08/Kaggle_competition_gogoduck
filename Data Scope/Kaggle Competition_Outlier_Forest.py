#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import IsolationForest


# In[3]:


train = pd.read_csv(r'C:\Users\yyang\Downloads\train.csv')


# In[9]:


train.head(10)


# In[11]:


train_vars = train.drop(columns = ['ID_code', 'target'])


# In[12]:


train_vars


# In[14]:


clf = IsolationForest(max_samples=100, random_state=42)
# table = pd.concat([input_table['Mean(ArrDelay)']], axis=1)
clf.fit(train_vars)
output_table = pd.DataFrame(clf.predict(train_vars))


# In[18]:


output_table.loc[output_table[0] == -1]


# In[ ]:


count()

