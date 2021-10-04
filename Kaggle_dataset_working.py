#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from IPython.core.display import display, HTML
import requests
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats
from sklearn import preprocessing
def retrieve_iMDB_links(pages, start=1):
    
    links = []
    for i in range(pages):
        url = f"https://www.imdb.com/search/title/?release_date=2018-01-01,2020-12-31&view=simple&count=100"
        response = requests.get(url)
        page = response.text
        soup = BeautifulSoup(page)
        find_urls = soup.find_all(class_="lister-item-index unbold text-primary")
        link_list = [i.findNext().findChildren()[0]["href"] for i in find_urls]
        links.extend(link_list)
        start += 10
    return links


# In[2]:


dataframe = pd.DataFrame(pd.read_csv('IMDb_movies.csv'))


# In[3]:


print(dataframe)


# In[4]:


iMDB_dataframe = dataframe.copy()


# In[5]:


iMDB_dataframe.drop_duplicates(subset=['title'],inplace=True)


# In[6]:


iMDB_dataframe.set_index('title',inplace=True)


# In[7]:


iMDB_dataframe_drop = iMDB_dataframe


# In[8]:


dataframe.info()


# In[9]:


iMDB_drop = iMDB_dataframe_drop[iMDB_dataframe_drop['avg_vote'].notna()]


# In[10]:


iMDB_drop = iMDB_drop[iMDB_drop['duration'].notna()]


# In[11]:


X, y = iMDB_drop[['duration']], iMDB_drop['avg_vote']


# In[12]:


lr = LinearRegression()
lr.fit(X, y)
print('R-squared: {:.4f}'.format(lr.score(X, y)))


# In[13]:



Genre_dataframe = pd.get_dummies(iMDB_drop['genre'])
Genre_dummies = pd.concat([iMDB_drop, Genre_dataframe], axis=1)


# In[14]:


genre_dummies = pd.concat([iMDB_drop, Genre_dataframe], axis=1)


# In[15]:


X, y = genre_dummies[['duration',
        'Music', 'Sci-Fi', 'Mystery', 'Sport', 'Family', 'War', 'Western', 'Musical', 'Documentary', 'Action', 'Fantasy', 'Horror', 'Biography', 'Drama', 'Comedy', 'Thriller', 'Animation', 'Crime', 'Adventure', 'History', 'Romance']], genre_dummies['avg_vote']
lr = LinearRegression()
lr.fit(X, y)
print('R-squared: {:.4f}'.format(lr.score(X, y)))


# In[16]:


Genre_dataframe = pd.Series(Genre_dummies['genre'])


# In[17]:


Genre_dataframe = pd.get_dummies(Genre_dataframe.apply(pd.Series).stack()).sum(level=0)


# In[18]:


dataframe_genres = pd.concat([genre_dummies, iMDB_drop], axis=1)


# In[19]:


X, y = dataframe_genres[['duration',
        'Music', 'Sci-Fi', 'Mystery', 'Sport', 'Family', 'War', 'Western', 'Musical', 'Documentary', 'Action', 'Fantasy', 'Horror', 'Biography', 'Drama', 'Comedy', 'Thriller', 'Animation', 'Crime', 'Adventure', 'History', 'Romance']], dataframe_genres['avg_vote']
lr = LinearRegression()
lr.fit(X, y)
print('R-squared: {:.4f}'.format(lr.score(X, y)))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.1, random_state=10)


# In[20]:


from sklearn.model_selection import train_test_split
X, y = dataframe_genres[['duration',
        'Music', 'Sci-Fi', 'Mystery', 'Sport', 'Family', 'War', 'Western', 'Musical', 'Documentary', 'Action', 'Fantasy', 'Horror', 'Biography', 'Drama', 'Comedy', 'Thriller', 'Animation', 'Crime', 'Adventure', 'History', 'Romance']], dataframe_genres['avg_vote']
X, X_test, y, y_test = train_test_split(X, y, test_size=.1, random_state=10)


# In[21]:


X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=10)
X, y = np.array(X), np.array(y) 
lr = LinearRegression()


# In[25]:


kf = KFold(n_splits=10, shuffle=True, random_state = 1)
cv_lr_r2s, cv_lr_reg_r2s = [], []

for train_ind, val_ind in kf.split(X,y):
    
    X_train, y_train = X[train_ind], y[train_ind]
    X_val, y_val = X[val_ind], y[val_ind] 
    
    #Linear Regression
    lr = LinearRegression()
    lr_reg = Ridge(alpha=1)

    lr.fit(X_train, y_train)
    cv_lr_r2s.append(lr.score(X_val, y_val))
    
    #Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lr_reg.fit(X_train_scaled, y_train)
    cv_lr_reg_r2s.append(lr_reg.score(X_val_scaled, y_val))

print('Simple regression scores: ', cv_lr_r2s)
print('Ridge scores: ', cv_lr_reg_r2s, '\n')

print(f'Simple mean cv R-squared: {np.mean(cv_lr_r2s):.3f} +- {np.std(cv_lr_r2s):.3f}')
print(f'Ridge mean cv R-squared: {np.mean(cv_lr_reg_r2s):.3f} +- {np.std(cv_lr_reg_r2s):.3f}')


# In[26]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[27]:


lr_reg.fit(X_train_scaled, y_train);


# In[28]:


test_set_pred = lr_reg.predict(X_test_scaled)


# In[29]:


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true)) 

print('MAE: {:.4f}'.format(mae(y_test, test_set_pred)))


# In[27]:


figure(figsize=(8, 6), dpi=80)
plt.scatter(test_set_pred, y_test, alpha=1, color='Yellow' )
plt.plot(np.linspace(0,10), np.linspace(0,10), color='Indigo')

plt.title('Model iMDB predictions vs actual iMDB ratings')
plt.xlabel('Model IMDb predictions')
plt.ylabel('Actual IMDb Rating');


# In[33]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

def build_and_compile_model():
  model = keras.Sequential([
      layers.Dense(256, activation='relu',input_shape=(23,)),
      layers.Dense(128, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(16, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


# In[34]:


print(X.shape)


# In[32]:


NN = build_and_compile_model()
NN.fit(X, y, validation_data=(X_val, y_val), epochs=5, batch_size=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




