
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


#Import Data
df_train = pd.read_csv('/Users/yuzhenhe/Desktop/train.csv')
df_test = pd.read_csv('/Users/yuzhenhe/Desktop/test.csv')


# In[7]:


#Drop Columns Target and ID_code
train_X = df_train.drop(columns=['target','ID_code'])


# In[8]:


#Build a dataset with the standardized values
from sklearn import preprocessing
names = train_X.columns
scaler = preprocessing.StandardScaler()
scaled_list = scaler.fit_transform(train_X)
scaled_df = pd.DataFrame(scaled_list, columns=names)


# In[9]:


#Transpose dataset
scaled_X_T = scaled_df.T
scaled_X_L = scaled_X_T.values.tolist()
group = df_train['target'].values.tolist()


# In[10]:


#Get the outliner of each column
outliner_index = []
outliner_count = []
outliner_unique = []
for k in scaled_X_L:
    p = [i for i, e in enumerate(k) if abs(e)>2.5 ]
    r = len(p)
    t = list(np.array(group)[p])
    outliner_index.append(p)
    outliner_count.append(r)
    outliner_unique = outliner_unique + p


# In[11]:


#Retain the unique outliner index
dic = {}
for i in outliner_index:
    for j in i:
        if j in dic:
            dic[j] += 1
        else:
            dic[j] = 1


# In[12]:


dic

