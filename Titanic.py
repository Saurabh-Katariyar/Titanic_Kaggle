
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


def main(train):
    
    import numpy as np 

    # delete the passenger id since it gives no indication on the data whatsoever 

    del train['PassengerId']
    # map Sex
    train.Sex=train.Sex.map({'male':1,'female':2})
    # use the name to get (mr.,ms. ... etc)
    import re 

    def search_pattern(index):
        return re.search(',.*[/.\b]',train.Name[index])[0][2:-1]

    train['Social_name']=[search_pattern(counter) for counter in range(train.shape[0]) ]

    # cleaning the things that the regex couldn't get 
    train.Social_name.replace({"""Mrs. Martin (Elizabeth L""":'Mrs',
                           'Mme':'Miss',
                           'Ms':'Miss',
                           'the Countess':'Countess',
                            'Mr. Carl':'Mr',
                            'Mlle':'Miss'},inplace=True)

    # mapping the values 
    for x in range(len(train.Social_name.unique())):
        a=train.Social_name.unique()[x]
        b=x

        train.Social_name.replace({a:b},inplace=True)


    # delete the name because we don't need it anymore 
    del train['Name']

    # fill na age values given (Pclass , Sex , Social_name)
    grouped=train.Age.groupby([train.Pclass,train.Sex,train.Social_name]).median()

    pclass=grouped.index.labels[0] ; sex=grouped.index.labels[1] ; social_name=grouped.index.labels[2]


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

    # from HERE to THERE is the same as putting inplace=True on the fillna but it seems that inplace doesn't work for no specific reason . . . 



    # map Embarked 
    train.Embarked=train.Embarked.map({'S':1,'C':2,'Q':3})

    # fill Embarked nans 
    train.Embarked.groupby(train.Embarked).count() # the max is 1 so we fill the nans with it 
    train.Embarked.fillna(1,inplace=True)

    # name the cabin with it's first letter 

    for x in range(len(train)):
        if pd.isnull(train.loc[x,'Cabin']):
            continue 
        else : 
            train.loc[x,'Cabin'] = train.loc[x,'Cabin'][0]

    # filling the nan cabin with a defaulted value 
    train.Cabin.fillna('N',inplace=True)

    # add dummies to the data and concating them to the origional dataset 
    train = pd.concat([train, pd.get_dummies(train.Cabin)], axis=1)

    # delete the nan values and the origional Cabin column 
    del train['N']
    del train['Cabin']
    

    


    # rounding the ages 

    train.Age=train.Age.values.round().astype(int)

     # i can't see any useful information from keeping the tickets so i will just delete it 

    del train['Ticket']

    # rounding the fares to give less unique numbers 

    train.Fare=train.Fare.round().astype(int)

    # Embarked means where did people get to the chip from (and that has no indication on the data in a logical manner so we delete it )
    del train['Embarked']
    return train


# In[3]:


df_train = pd.read_csv('train.csv')
df_train = main(df_train)

del df_train['T']
model_training = df_train.loc[:,df_train.columns!='Survived']
model_testing = df_train.loc[:,'Survived']


# In[4]:


df_test = pd.read_csv('test.csv')

df_test.Fare.fillna(df_test.Fare.median(skipna = True),inplace = True)

df_test = main(df_test)
                    


# In[5]:


from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()

the_model = logistic.fit(model_training,model_testing)
results = the_model.predict(df_test)


# In[6]:


final = pd.read_csv('test.csv',usecols = ['PassengerId'])
final['Survived'] = results

final.to_csv('final.csv')

