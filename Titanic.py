
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


train.isnull().sum()


# In[4]:


del train['Embarked']
del train['PassengerId']


# In[5]:


# map sex
train.Sex = train.Sex.map({'male':1,'female':2})


# In[6]:


import re

def search_pattern (index):
    return re.search(',.*[/.\b]',train.Name[index])[0][2:-1]

train['Social_name'] = [search_pattern(counter) for counter in range(train.shape[0])]


# In[7]:


train.Social_name.unique()


# In[8]:


train.Social_name.replace({'Mme':'Miss','Mr. Carl':'Mr','Ms':'Miss',"""Mrs. Martin (Elizabeth L""":'Mrs','Mlle':'Miss','the Countess':'Countess'},inplace=True)


# In[9]:


train.Social_name.unique()


# In[10]:


for index in range(len(train.Social_name.unique())):
    
    a = train.Social_name.unique()[index]
    train.Social_name.replace({a:index},inplace = True)
    
del train['Name']


# In[11]:


train.head()


# In[12]:


train.Age.groupby([train.Pclass,train.Sex,train.Social_name]).median()


# In[13]:


grouped = train.Age.groupby([train.Pclass, train.Sex, train.Social_name]).median()
pclass = grouped.index.labels[0]
sex = grouped.index.labels[1]
social_name = grouped.index.labels[2]


# In[14]:


for counter in range(len(grouped.index.labels[1])):
    # HERE
    train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
              (train.Sex==train.Sex.unique()[sex[counter]]) &
              (train.Social_name==train.Social_name.unique()[social_name[counter]])),
              'Age']=\
    train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
              (train.Sex==train.Sex.unique()[sex[counter]]) &
              (train.Social_name==train.Social_name.unique()[social_name[counter]])),
              'Age'].fillna(value=grouped.values[counter])
    # THERE


# In[15]:


train.head(2)


# In[16]:


train.Cabin.unique()


# In[18]:


for x in range(len(train)):
    if pd.isnull(train.loc[x,'Cabin']):
        continue
        
    else:
        train.loc[x,'Cabin'] = train.loc[x,'Cabin'][0]
        
train.Cabin.fillna('N',inplace=True)


# In[19]:


train = pd.concat([train, pd.get_dummies(train.Cabin)],axis = 1)
train


# In[22]:


del train['Cabin']
del train['N']


# In[23]:


len(train.Ticket.unique())


# In[24]:


del train['Ticket']


# In[27]:


train.isnull().sum()

