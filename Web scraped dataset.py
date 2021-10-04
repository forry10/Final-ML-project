#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install bs4==0.0.1')


# In[ ]:


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


# In[ ]:


iMDB_links = retrieve_iMDB_links(1)


# In[ ]:


def parse_iMDB_features(link):
    
    base_url = "https://www.imdb.com"

    url = base_url + link

    response = requests.get(url)
    page = response.text
    soup = BeautifulSoup(page)

    headers = [
        "movie title",
        "imdb rating",
        "imdb raters",
        "BBFC",
        "genres",
        "director",
        "writer",
        "stars",
        "country",
        "language",
        "release date",
        "budget",
        "opening weekend",
        "cumulative worldwide",
        "production companies",
        "runtime (min)",
    ]

    
    try:
        title = (
            soup.find(class_="title_wrapper").find("h1").text.split("\xa0")[0]
        )
    except:
        title = None

    
    try:
        rating_10 = float(soup.find(class_="ratingValue").span.text)
    except:
        rating_10 = None

    
    try:
        raters = int(
            soup.find(class_="ratingValue")
            .strong["title"]
            .split()[3]
            .replace(",", "")
        )
    except:
        raters = None

    
    BBFC_options = [
        "U",
        "PG",
        "12A",
        "12",
        "15",
        "18",
        "R18"
    ]
    try:
        BBFC = soup.find(class_="subtext").text
        BBFC = BBFC.split("\n")[1].strip()
        if BBFC not in BBFC_options:
            BBFC = None
    except:
        BBFC = None

    
    try:
        genre = soup.find("h4", text=re.compile("Genre")).findParent().text
        genre = [
            ele.strip().replace("\xa0|", "") for ele in genre.split("\n")[2:-1]
        ]
    except:
        genre = None

    
    try:
        director = (
            soup.find_all(class_="credit_summary_item")[0]
            .text.split("\n")[-1]
            .strip()
        )
    except:
        director = None

    
    try:
        writer = (
            soup.find_all(class_="credit_summary_item")[1]
            .text.split("\n")[2]
            .split(",")[0]
        )
        writer = writer.split("(")[0].strip()
    except:
        writer = None

    
    try:
        stars_unclean = (
            soup.find_all(class_="credit_summary_item")[2]
            .text.split("\n")[2]
            .split(",")
        )
        stars = [i.replace("|", "").strip() for i in stars_unclean]
    except:
        stars = None

    
    try:
        country = soup.find("h4", text=re.compile("Country")).findNext().text
    except:
        country = None

    
    try:
        language = soup.find("h4", text=re.compile("Language")).findNext().text
    except:
        language = None

    
    try:
        release_date = (
            soup.find("h4", text=re.compile("Release Date")).findParent().text
        )
        release_date = (
            release_date.split("\n")[1].split(":")[1].split("(")[0].strip()
        )
        release_date = datetime.strptime(release_date, "%d %B %Y").date()
    except:
        release_date = None

    
    try:
        budget = soup.find("h4", text=re.compile("Budget")).findParent().text
        budget = budget.split("\n")[1].split(":")[1]
        budget = money_to_int(budget)
    except:
        budget = None

    
    try:
        opening_weekend = (
            soup.find("h4", text=re.compile("Opening Weekend"))
            .findParent()
            .text
        )
        opening_weekend = (
            opening_weekend.split("\n")[1].split(":")[1].strip(" ,")
        )
        opening_weekend = money_to_int(opening_weekend)
    except:
        opening_weekend = None

   

   
    try:
        worldwide = (
            soup.find("h4", text=re.compile("Cumulative Worldwide"))
            .findParent()
            .text
        )
        worldwide = worldwide.split(":")[1].strip()
        worldwide = money_to_int(worldwide)
    except:
        worldwide = None

    
    try:
        production_co = (
            soup.find("h4", text=re.compile("Production Co")).findParent().text
        )
        production_co = production_co.split("\n")[2].strip()
        production_co = [co.strip() for co in production_co.split(",")]
    except:
        production_co = None

    
    try:
        runtime = soup.find("h4", text=re.compile("Runtime")).findParent().text
        runtime = int(runtime.split("\n")[2].split(" ")[0])
    except:
        runtime = None

    data_list = [
        title,
        rating_10,
        raters,
        BBFC,
        genre,
        director,
        writer,
        stars,
        country,
        language,
        release_date,
        budget,
        opening_weekend,
        worldwide,
        production_co,
        runtime,
    ]

    movie_dict = dict(zip(headers, data_list))

    return movie_dict



# In[ ]:


print(iMDB_links)


# In[ ]:


count = 0
iMDB_converted_data = []
for link in movie_links:
    iMDB_converted_data.append(parse_iMDB_features(link))
    count += 1
    if count % 50 == 0:
        with open('IMDb_data_t.pickle', 'wb') as to_write:
            pickle.dump(iMDB_converted_data, to_write)


# In[ ]:


iMDB_data = pd.DataFrame(iMDB_converted_data)


# In[ ]:





# In[ ]:


dataframe = pd.DataFrame(pd.read_pickle('IMDb_data_test.pickle'))


# In[ ]:


print(dataframe)


# In[ ]:



iMDB_dataframe = dataframe.copy()


# In[ ]:


iMDB_dataframe.drop_duplicates(subset=['movie title'],inplace=True)


# In[ ]:


iMDB_dataframe.set_index('movie title',inplace=True)


# In[ ]:


iMDB_dataframe_drop = iMDB_dataframe


# In[ ]:


dataframe.info()


# In[ ]:


sns.pairplot(dataframe, height=2, aspect=2.15)
plt.savefig('pairplot.png');


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


iMDB_drop = iMDB_dataframe_drop[iMDB_dataframe_drop['imdb rating'].notna()]


# In[ ]:


iMDB_drop = iMDB_drop[iMDB_drop['runtime (min)'].notna()]


# In[ ]:


X, y = iMDB_drop[['runtime (min)']], iMDB_drop['imdb rating']


# In[ ]:


lr = LinearRegression()
lr.fit(X, y)
print('R-squared: {:.4f}'.format(lr.score(X, y)))


# In[ ]:


BBFC_dataframe = pd.get_dummies(iMDB_drop['BBFC'])


# In[ ]:


BBFC_dummies = pd.concat([iMDB_drop, BBFC_dataframe], axis=1)


# In[ ]:


X, y = BBFC_dummies[['runtime (min)',
        "PG",
        "12",
        "15",
        "18"]], BBFC_dummies['imdb rating']
lr = LinearRegression()
lr.fit(X, y)
print('R-squared: {:.4f}'.format(lr.score(X, y)))


# In[ ]:


Genre_dataframe = pd.Series(BBFC_dummies['genres'])


# In[ ]:


Genre_dataframe = pd.get_dummies(Genre_dataframe.apply(pd.Series).stack()).sum(level=0)


# In[ ]:


dataframe_genresandBBFC = pd.concat([BBFC_dummies, Genre_dataframe], axis=1)


# In[ ]:


dataframe_genresandBBFC.drop(labels=['budget', 'release date'],axis=1,inplace=True)


# In[ ]:


X, y = dataframe_genresandBBFC[['runtime (min)',
        "PG",
        "12A",
        "12",
        "15",
        "18",'Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Family','Fantasy','History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western']], dataframe_genresandBBFC['imdb rating']
lr = LinearRegression()
lr.fit(X, y)
print('R-squared: {:.4f}'.format(lr.score(X, y)))


# In[ ]:


iMDB_drop[iMDB_drop['director']==''].index.tolist()


# In[ ]:


empty_dir = iMDB_drop[iMDB_drop['director']==''].index.values.tolist()


# In[ ]:


def directors_list(directors):
   
    if "," in directors:
        return [name.strip() for name in directors.split(",")]
    else:
        return [directors]


def remove_paren(directors):
   
    director_list = []
    for director in directors:
        if "(" in director:
            director_clean = director.split("(")[0].strip()
            director_list.append(director_clean)
        else:
            director_list.append(director)
    return director_list


# In[ ]:


iMDB_drop['director'] = iMDB_drop['director'].apply(lambda x: directors_list(x))


# In[ ]:


iMDB_drop['director'] = iMDB_drop['director'].apply(lambda x: remove_paren(x))


# In[ ]:


dataframe_director_exploded = iMDB_drop.explode('director')


# In[ ]:


director_10 = dataframe_director_exploded['director'].value_counts()[:10].index.tolist()


# In[ ]:


def top_directors(directors):
    director_list = []
    for director in directors:
        if director in director_10:
            director_list.append(director)
    return director_list


# In[ ]:


iMDB_drop['top directors'] = iMDB_drop['director'].apply(lambda x: top_directors(x))


# In[ ]:


director_dataframe = pd.get_dummies(iMDB_drop['top directors'].apply(pd.Series).stack()).sum(level=0)


# In[ ]:


dataframe_director_model = pd.concat([dataframe_genresandBBFC, director_dataframe], axis=1)


# In[ ]:


dataframe_director_model.drop(labels=['genres','director','writer','stars','country','language','production companies'],axis=1,inplace=True)
# Let's take out the columns we're not using


# In[ ]:


dataframe_director_model.replace(np.nan,0,inplace=True)


# In[ ]:


X, y = dataframe_director_model.iloc[:,2:], dataframe_director_model['imdb rating']
lr = LinearRegression()
lr.fit(X, y)
print('R-squared: {:.4f}'.format(lr.score(X, y)))


# In[ ]:


writer_dataframe = pd.get_dummies(iMDB_drop['writer'])


# In[ ]:



writer_10 = iMDB_drop.writer.value_counts()[:10].index.tolist()


# In[ ]:


dataframe_writer_model = pd.concat([dataframe_director_model, writer_dataframe[writer_10]],axis=1)


# In[ ]:


X, y = dataframe_writer_model.iloc[:,2:], dataframe_writer_model['imdb rating']
lr = LinearRegression()
lr.fit(X, y)
print('R-squared: {:.4f}'.format(lr.score(X, y)))


# In[ ]:


from sklearn.model_selection import train_test_split
X, y = dataframe_writer_model.iloc[:,2:,], dataframe_writer_model['imdb rating']
X, X_test, y, y_test = train_test_split(X, y, test_size=.1, random_state=10)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.1, random_state=10)


# In[ ]:


X, y = dataframe_writer_model.iloc[:,2:,], dataframe_writer_model['imdb rating']


# In[ ]:


print(dataframe_writer_model)


# In[ ]:


dataframe_writer_model.info


# In[ ]:


X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=10)
X, y = np.array(X), np.array(y) 
lr = LinearRegression()


# In[ ]:


kf = KFold(n_splits=10, shuffle=True, random_state = 1)
cv_lr_r2s, cv_lr_reg_r2s = [], []

for train_ind, val_ind in kf.split(X,y):
    
    X_train, y_train = X[train_ind], y[train_ind]
    X_val, y_val = X[val_ind], y[val_ind] 
    
    lr = LinearRegression()
    lr_reg = Ridge(alpha=1)

    lr.fit(X_train, y_train)
    cv_lr_r2s.append(lr.score(X_val, y_val))
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lr_reg.fit(X_train_scaled, y_train)
    cv_lr_reg_r2s.append(lr_reg.score(X_val_scaled, y_val))

print('Simple regression scores: ', cv_lr_r2s)
print('Ridge scores: ', cv_lr_reg_r2s, '\n')

print(f'Simple mean cv R-squared: {np.mean(cv_lr_r2s):.3f} +- {np.std(cv_lr_r2s):.3f}')
print(f'Ridge mean cv R-squared: {np.mean(cv_lr_reg_r2s):.3f} +- {np.std(cv_lr_reg_r2s):.3f}')


# In[ ]:


alphas = np.logspace(-5, 5, 15)

ridge = RidgeCV(alphas=alphas, normalize = True)

ridge.fit(X_train, y_train)


# In[ ]:


ridge.coef_


# In[ ]:


ridge.alpha_


# In[ ]:


print(ridge.score(X_val, y_val))


# In[ ]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


lr_reg.fit(X_train_scaled, y_train);


# In[ ]:


test_set_pred = lr_reg.predict(X_test_scaled)


# In[ ]:


figure(figsize=(8, 6), dpi=80)
plt.scatter(test_set_pred, y_test, alpha=1, color='Red' )
plt.plot(np.linspace(0,10), np.linspace(0,10), color='Green')

plt.title('Model iMDB predictions vs actual iMDB ratings')
plt.xlabel('Model IMDb predictions')
plt.ylabel('Actual IMDb Rating');


# In[ ]:


print('Ridge R-squared: {:.4f}'.format(r2_score(y_test, test_set_pred))) 


# In[ ]:


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true)) 

print('MAE: {:.4f}'.format(mae(y_test, test_set_pred)))


# In[ ]:


def diagnostic_plot(x, y):
    plt.figure(figsize=(20,5))

    pred = x

    plt.subplot(1, 3, 1)
    plt.scatter(x,y)
    plt.plot(x, pred, color='red',linewidth=1)
    plt.title("Regression fit")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.subplot(1, 3, 2)
    res = y - pred
    plt.scatter(pred, res)
    plt.title("Residual plot")
    plt.xlabel("prediction")
    plt.ylabel("residuals")
    
    plt.subplot(1, 3, 3)
    
    stats.probplot(res, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot");


# In[ ]:


diagnostic_plot(test_set_pred, y_test)


# In[ ]:


get_ipython().system('pip install xgboost')
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model


# In[ ]:


# XGboost
xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X,
         y)


# In[ ]:


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)


# In[ ]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

def build_and_compile_model():
  model = keras.Sequential([
      layers.Dense(256, activation='relu',input_shape=(46,)),
      layers.Dense(128, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(16, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


# In[ ]:


# normalise here


NN = build_and_compile_model()
NN.fit(X, y, validation_data=(X_val, y_val), epochs=50, batch_size=1)


# In[ ]:


def diagnostic_plot(x, y):
    plt.figure(figsize=(20,5))

    pred = x

    plt.subplot(1, 3, 1)
    plt.scatter(x,y)
    plt.plot(x, pred, color='red',linewidth=1)
    plt.title("Regression fit")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.subplot(1, 3, 2)
    res = y - pred
    plt.scatter(pred, res)
    plt.title("Residual plot")
    plt.xlabel("prediction")
    plt.ylabel("residuals")
    
    plt.subplot(1, 3, 3)
    
    stats.probplot(res, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot");


# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=807fde48-b748-4f76-9f2e-6abbdbc68210' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
