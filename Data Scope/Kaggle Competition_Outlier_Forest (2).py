#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import IsolationForest


# In[2]:


#import training dataset
train = pd.read_csv(r'C:\Users\yyang\Downloads\train.csv')


# In[3]:


train.head(10)


# In[4]:


#'ID_code', 'target' are not independent variavbles and they will not envolve to the outlier filtering
train_vars = train.drop(columns = ['ID_code', 'target'])


# In[5]:


train_vars


#  Isolation Forest, like any tree ensemble method, is built on the basis of decision trees. In these trees, partitions are created by first randomly selecting a feature and then selecting a random split value between the minimum and maximum value of the selected feature.
# 
# In principle, outliers are less frequent than regular observations and are different from them in terms of values (they lie further away from the regular observations in the feature space).
# 
# More reference: https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e

# In[6]:


# Isolation Forest to find outliers
clf = IsolationForest(max_samples=100, random_state=42)
# table = pd.concat([input_table['Mean(ArrDelay)']], axis=1)
clf.fit(train_vars)
output_table = pd.DataFrame(clf.predict(train_vars))


# In[7]:


# select the rows that are outliers 
# predict of Isolation Forest equals to -1 --> outlier; 1 --> normal
output_table.loc[output_table[0] == -1]

