#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# # Inserting fake and real dataset

# In[2]:


df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")


# In[3]:


df_fake.head()


# In[4]:


df_true.head()


# In[5]:


df_fake["class"] = 0
df_true["class"] = 1


# In[6]:


df_fake.shape ,df_true.shape


# In[7]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[9]:


#Removing  last 10 rows from both the dataset, for manual testing


# In[10]:


df_fake.shape ,df_true.shape


# In[11]:


df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1


# In[12]:


df_fake_manual_testing.head(10)


# In[13]:


df_true_manual_testing.head(10)


# In[14]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")


# In[15]:


#Merging the main fake and true dataframe


# In[16]:


df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)


# In[17]:


df_marge.columns


# # "title","subject" and "date" columns is not required for detecting the fake news,  hence drop the columns.

# In[18]:


df = df_marge.drop(["title", "subject","date"], axis = 1)


# In[19]:


df.isnull().sum()


# ###Randomly shuffling the dataframe

# In[20]:


df = df.sample(frac = 1)
df.head()


# In[21]:


df.columns


# ###Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr and links.

# In[22]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[23]:


df["text"] = df["text"].apply(wordopt)


# #Defining dependent and independent variable as x and y 

# In[24]:


x = df["text"]
y = df["class"]


# # Splitting the dataset into training set and testing set

# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# # Convert text to vectors

# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[27]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# #1.Logisitc Regression

# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[30]:


pred_lr=LR.predict(xv_test)


# In[31]:


LR.score(xv_test, y_test)


# In[32]:


print(classification_report(y_test, pred_lr))


# #2.Desicion Tree classification

# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[36]:


pred_dt = DT.predict(xv_test)


# In[37]:


DT.score(xv_test, y_test)


# In[38]:


print(classification_report(y_test, pred_dt))


# In[ ]:


#so we got a higher percentage with Decision Tree classifier method

