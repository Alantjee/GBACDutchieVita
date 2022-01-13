#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from time import time
import os
import pydotplus
from IPython.display import Image

from typing import Union, List, Tuple  # for type hints in function definitions
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold
from sklearn.inspection import plot_partial_dependence, partial_dependence

from six import StringIO
from matplotlib import cm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
#Standard libraries for data analysis:
from datetime import date, datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, skew
from scipy import stats
import statsmodels.api as sm
# sklearn modules for data preprocessing:
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#sklearn modules for Model Selection:
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#sklearn modules for Model Evaluation & Improvement:
    
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer, recall_score, log_loss
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score,confusion_matrix

#Standard libraries for data visualization:
import seaborn as sn
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib 
get_ipython().run_line_magic('matplotlib', 'inline')
color = sn.color_palette()
import matplotlib.ticker as mtick
from IPython.display import display
pd.options.display.max_columns = None
from pandas.plotting import scatter_matrix
from sklearn.metrics import roc_curve
#Miscellaneous Utilitiy Libraries:
    
import random
import os
import re
import sys
import timeit
import string

from datetime import datetime
import time
from dateutil.parser import parse
import joblib


# In[2]:


#read in data 
df2 = pd.read_csv('C:\Documents\Anagrafica_ClubQ8.csv', sep = ';', decimal=",")

#df2[['YEAR_JOINED','TIME_JOINED']] = df2.DATA_BATTESIMO.str.split(" ",expand=True,)

#df2 = df2.drop(columns="DATA_BATTESIMO")

#rename columns
df2.columns = ['CUSTOMER_CODE', 'SEX', 'DOB', 'REGION', 'PROVINCE', 'CITY', 'TYPE_OF_CARD','DATA_BATTESIMO' ,'TOTAL_POINTS']

df2


# In[3]:


#read in the transaction dataset
df3 = pd.read_csv('C:\Documents\Rifornimenti_Carburante_ClubQ8.csv', sep = ';', decimal=",")

#split year and time
df3[['REQUEST_YEAR','REQUEST_TIME']] = df3.DATA_OPERAZIONE.str.split(" ",expand=True,)

#drop original column
df3 = df3.drop(columns="DATA_OPERAZIONE")
#rename columns
df3.columns = ['CUSTOMER_CODE', 'STATION_CODE', 'PRODUCT', 'SALES_MODALITY', 'LITRES', 'LOYALTY_POINTS_AWARDED','REQUEST_YEAR','REQUEST_TIME']

df3


# In[4]:


#create a copy of customer id's to keep the customer id's that are only in the transaction datasets when merging
df3['CUSTOMER_CODE2'] = df3['CUSTOMER_CODE']


# In[5]:


#take the two columns with customer id's to merge them with the customer database
df3 = df3[['CUSTOMER_CODE', 'CUSTOMER_CODE2']]

df3


# In[6]:


#get a list of unique customer id's that made a transaction
df3 = df3.drop_duplicates()
df3


# In[7]:


#merge the list with customers that made a transaction with the customer database
dfnew = pd.merge(df2, df3, how = 'left')
dfnew


# In[8]:


#fill na's with 0 to prepare for variable recoding
dfnew['CUSTOMER_CODE2'] = dfnew['CUSTOMER_CODE2'].fillna(0.0)
#create a variable churn that is equal to 1 when the customer has a customer id that is not in the list of customers that made a transaction
dfnew['Churn'] = np.where(dfnew['CUSTOMER_CODE2'] == 0.0, 1, 0)   

dfnew


# In[9]:


#create a function that computes the age based on comparing the current date with the date of birth
def age(born):
    born = datetime.strptime(born, "%Y-%m-%d %H:%M:%S").date()
    today = date.today()
    return today.year - born.year - ((today.month, 
                                      today.day) < (born.month, 
                                                    born.day))

#construct a variable that computes the age for each customer
dfnew['Age'] = dfnew['DOB'].apply(age)


# In[10]:


dfnew = dfnew[dfnew['DATA_BATTESIMO'].notna()]
#similar function to before to compute the duration of membership for each customer by subracting the date of joining from the current date
def age(born):
    born = datetime.strptime(str(born), "%Y-%m-%d %H:%M:%S").date()
    today = date.today()
    return today.year - born.year - ((today.month, 
                                      today.day) < (born.month, 
                                                    born.day))
#calculate number of years nad convert to months 
dfnew['Tenure in years'] = dfnew['DATA_BATTESIMO'].apply(age)
dfnew['Tenure in months'] = round(dfnew['Tenure in years']*12)


# In[11]:


dfnew.dropna(how='any', inplace=True)


# In[12]:


def encode_onehot(series: pd.Series, drop_last: bool = False
                  ) -> Union[pd.Series, pd.DataFrame]:
    values = series.unique()
    if drop_last:
        values = values[:-1]
    return pd.concat(((series == val).rename(val) 
                      for val in values
                      ), 
                     axis=1
                     ).squeeze()


# In[13]:


dfnew['Male'] = encode_onehot(dfnew['SEX'], drop_last=True).astype(int)
dfnew[['physicalcard','virtualcard']] = encode_onehot(dfnew['TYPE_OF_CARD'], drop_last=True).astype(int)


# In[ ]:



dfnew[(np.abs(stats.zscore(dfnew['TOTAL_POINTS'])) < 6)]


# In[ ]:



dfnew[['TOTAL_POINTS','Tenure in months','Age']] = StandardScaler().fit_transform(dfnew[['TOTAL_POINTS','Tenure in months','Age']])


# In[14]:


# Features
X = pd.concat([dfnew['Male'],dfnew['physicalcard'],dfnew['Age'],dfnew['TOTAL_POINTS'],dfnew['virtualcard'], dfnew['Tenure in months']
], axis=1, keys=['gender' ,'physicalcard','age','points','virtualcard','tenure'])
print("Shape of features' set:", X.shape)

# Response (observed number of claims)
y = dfnew['Churn']
print("Shape of response' set:", y.shape)


# In[15]:


def filter_std(series: pd.Series, n_std: float):
    m = (series - series.mean()).abs() > n_std * series.std()
    return series[m]


# In[16]:


totalpoints = dfnew['TOTAL_POINTS']
totalpoints_outliers = filter_std(totalpoints, n_std=6)

print(f"Identified {len(totalpoints_outliers)} outliers "
      f"({len(totalpoints_outliers) / len(dfnew) * 100:02.2f}% of observations)"
      )


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=5, stratify=y)


# In[19]:


def poisson_deviance(y, y_pred):
    
    y = np.array(y)
    y_pred = np.array(y_pred)
    
    nlogn = np.empty(y.size)
    dev = np.empty(y.size)
    
    for i in range(y.size):
        if y[i] == 0:
            nlogn[i] = 0
        else:
            nlogn[i] = y[i] * np.log(y[i] / y_pred[i])
            
        dev[i] = 2 * (nlogn[i] - (y[i] - y_pred[i]))
        
    return(dev.sum())


# In[20]:


def mean_poisson_deviance(y, y_pred):
    return(poisson_deviance(y, y_pred)/y.size)


# In[21]:


global_lambda = y_train.sum() / y_train.size
print("Global claim frequency:", global_lambda)


# In[22]:


# Prediction
y_pred_train = np.repeat(global_lambda, y_train.size)
y_pred_test = np.repeat(global_lambda, y_test.size)

# Metrics
print("Mean Poisson deviance on training set:", mean_poisson_deviance(y_train, y_pred_train))
print("Mean Poisson deviance on testing set:", mean_poisson_deviance(y_test, y_pred_test))


# In[23]:


model = LogisticRegression()


# In[24]:


model.fit(X_train, y_train)


# In[25]:


# In[26]:


model.coef_


# In[27]:


# Prediction
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# In[ ]:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#Generate the confusion matrix
cf_matrix = confusion_matrix(y_pred_test.round(), y_test)

print(cf_matrix)

accuracy = accuracy_score(y_test, y_pred_test.round())

print(accuracy)

recall = recall_score(y_pred_test.round(), y_test, average=None)

print(recall)

precision = precision_score(y_pred_test.round(), y_test, average = None)

print(precision)



# In[ ]:


print(classification_report(y_test, y_pred_test.round()))


