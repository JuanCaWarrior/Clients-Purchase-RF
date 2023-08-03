#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


bel = pd.read_csv('bel.csv',sep=';')


# In[7]:


bel.info()


# In[8]:


bel.describe()


# In[9]:


bel.head()


# In[10]:


sns.pairplot(bel,hue='PEDIDO',palette='Set1')


# In[11]:


plt.figure(figsize=(10,6))
bel[bel['CLASIFICACION']==1]['ESTADO'].hist(alpha=0.5,color='RED',
                                              bins=30,label='CLASIFICACION=1')
bel[bel['CLASIFICACION']==2]['ESTADO'].hist(alpha=0.5,color='GRAY',
                                              bins=30,label='CLASIFICACION=2')
bel[bel['CLASIFICACION']==3]['ESTADO'].hist(alpha=0.5,color='YELLOW',
                                              bins=30,label='CLASIFICACION=3')


plt.legend()
plt.xlabel('ESTADO')


# In[12]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# In[13]:


plt.figure(figsize=(10,6))
bel[bel['PEDIDO']==0]['CLASIFICACION'].hist(alpha=0.5,color='RED',
                                              bins=30,label='PEDIDO=0')
bel[bel['PEDIDO']==1]['CLASIFICACION'].hist(alpha=0.5,color='BLUE',
                                              bins=30,label='PEDIDO=1')

plt.legend()
plt.xlabel('CLASIFICACION')


# In[14]:


plt.figure(figsize=(10,6))
bel[bel['PEDIDO']==0]['ESTADO'].hist(alpha=0.5,color='RED',
                                              bins=30,label='PEDIDO=0')
bel[bel['PEDIDO']==1]['ESTADO'].hist(alpha=0.5,color='BLUE',
                                              bins=30,label='PEDIDO=1')

plt.legend()
plt.xlabel('ESTADO')


# In[15]:


cat_zone = ['ZONA']
cat_month = ['Mes']


# In[16]:


data = pd.get_dummies(bel, columns = cat_zone)


# In[17]:


final_data = pd.get_dummies(data, columns = cat_month)


# In[18]:


final_data.info()


# In[19]:


final_data.head(10)


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X = final_data.drop(['PEDIDO', 'ID'],axis=1)
y = final_data['PEDIDO']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


rfc = RandomForestClassifier(n_estimators=600)


# In[24]:


rfc.fit(X_train,y_train)


# In[25]:


predictions = rfc.predict(X_test)


# In[26]:


from sklearn.metrics import classification_report,confusion_matrix


# In[27]:


print(classification_report(y_test,predictions))


# In[28]:


print(confusion_matrix(y_test,predictions))


# In[32]:


a = rfc.predict(X)


# In[34]:


np.savetxt("ans.csv", a, delimiter=",")

