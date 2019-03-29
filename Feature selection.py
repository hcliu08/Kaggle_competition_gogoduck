#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold


# In[5]:


# load data
df = pd.read_csv('train.csv')
X = df.drop(columns = ['target', 'ID_code'])
y = df.target


# In[6]:


#split data
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0)


# In[19]:


#importance of the feature
clf = RandomForestClassifier(random_state = 0, n_jobs = 1).fit(X_train, y_train)
perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)
weight = eli5.show_weights(perm, feature_names = X_test.columns.tolist(), top=100)


# In[44]:


weight


# In[119]:


#select feature
df_train = df.loc[:199999,['var_81','var_26', 'var_44' , 'var_110','var_109','var_190', 'var_78','var_21', 'var_1', 'var_99', 
                    'var_133','var_166','var_34','var_148','var_122','var_139','var_164','var_12', 'var_165','var_119', 
                    'var_76','var_53','var_107','var_151','var_187','var_2','var_184','var_177','var_192','var_145', 
                    'var_57','var_155','var_134','var_40','var_89','var_180','var_157','var_194','var_90','var_22','var_83', 
                    'var_75','var_131','var_58','var_98','var_179','var_176','var_130','var_23','var_118','var_158','var_68', 
                    'var_174','var_97','var_94','var_91','var_69','var_93','var_33','var_62','var_80','var_128','var_86', 
                    'var_125','var_175','var_16']]


# In[120]:


df_train.head()


# In[91]:


#split new dataset
train_x,test_x,train_y,test_y = train_test_split(df_train,y,test_size=0.2,random_state=0)


# In[92]:


target_count=y.value_counts()
print('Class 0:',target_count[0])
print('Class 1:',target_count[1])
print('Proportion:',round(target_count[0]/target_count[1],2),': 1')


# In[93]:


target_count.plot(kind='bar',title='Count (target)');
plt.show()


# In[94]:


#merge target and X
df_train=train_x.join(train_y)
# Class count
count_class_0,count_class_1=y.value_counts()
# Divide by class
df_class_0=df_train[y == 0]
df_class_1=df_train[y == 1]


# In[95]:


df_class_1_over.head()


# In[101]:


#under sampling
df_class_0_under=df_class_0.sample(count_class_1)
df_test_under=pd.concat([df_class_0_under,df_class_1],axis=0)
print('Random under-sampling:')
print(df_test_under.target.value_counts())
df_test_under.target.value_counts().plot(kind='bar',title='Count (target)');


# In[100]:


# oversampling
df_class_1_over=df_class_1.sample(count_class_0 , replace = True)
df_test_over=pd.concat([df_class_1_over,df_class_0],axis=0)
print('Random over-sampling:')
print(df_test_over.target.value_counts())
df_test_over.target.value_counts().plot(kind='bar',title='Count (target)');


# In[ ]:


#K fold and model

