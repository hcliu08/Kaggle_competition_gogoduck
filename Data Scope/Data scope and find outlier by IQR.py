
# coding: utf-8

# In[1]:


# import
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


# upload data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[7]:


# view the data in same scale
df_train[df_train.dtypes[(df_train.dtypes=="float64")|(df_train.dtypes=="int64")]
                        .index.values].hist(figsize=[20,20])


# In[41]:


# view the outlier for each varible
df_train[df_train.dtypes[(df_train.dtypes=="float64")|(df_train.dtypes=="int64")]
                        .index.values].boxplot(figsize=[40,40])


# In[16]:


# split the target and X 
train_X = df_train.drop(columns=['target','ID_code'])
train_X_T = train_X.T
train_X_L = train_X_T.values.tolist()
train_y = [df_train.target.T]


# In[27]:


# Use IQR to find outlier for each varible
O_l = []
for i in range(len(train_X_L)):
    quartile_1 , quartile_3 = np.percentile(train_X_L[i],[25,75])
    iqr = quartile_3-quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    outlier = list(np.where((train_X_L[i] > upper_bound) | (train_X_L[i] < lower_bound)))
    
    O_l += outlier


# In[32]:


# count the and list the outlier
dic = {}
for i in O_l:
    for j in i:
        if j in dic:
            
            dic[j] += 1
        else:
            dic[j] = 1


# In[34]:


len(dic)

