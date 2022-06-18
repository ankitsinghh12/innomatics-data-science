#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import all the required Libraries

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.svm import SVC
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('fast')
sns.set_style('whitegrid')


# **What is the type of machine learning problem at hand? (Supervised or Unsupervised)**
# 
# It is a supervised learning problem and supervised learning uses labeled input and output data, while an unsupervised learning algorithm does not. ... Unsupervised learning models, in contrast, work on their own to discover the inherent structure of unlabeled data and this is a labeled data so we have labeled data so we are using supervised learnig. What is the category of the machine learning problem at hand? (Classification or Regression?) Why? This ml problem is regression problem
# 
# The main difference between Regression and Classification algorithms that Regression algorithms are used to predict the continuous values such as price, salary, age, etc. and Classification algorithms are used to predict/Classify the discrete values such as Male or Female, True or False, Spam or Not Spam.

# In[16]:


data=pd.read_csv(r'C:\Users\ani\Downloads\data.csv')#Read the csv file
#This dataset represents the characteristics of x1, x2, y


# In[17]:


data.head()


# In[18]:


data.shape


# In[19]:


data.describe()


# In[20]:



data.info()


# In[21]:


data.dropna(inplace=True)


# In[22]:


data.isnull().sum()


# In[23]:


data.shape


# In[25]:


data["B"].value_counts()


# In[28]:


x=data.drop("B",axis=1)
y=data.B


# In[29]:


sns.heatmap(data)


# In[30]:


corrmat=data.corr()
corrmat


# In[31]:


plt.figure(figsize=(20,20), dpi=100)
sns.heatmap(corrmat,annot=True)
plt.show()


# A heatmap is a data visualization technique that uses color to show how a value of interest changes depending on the values of two other variables.
# 
# For example, you could use a heatmap to understand how air pollution varies according to the time of day across a set of cities.
# 
# Another, perhaps more rare case of using heatmaps is to observe human behavior - you can create visualizations of how people use social media, how their answers on surveys changed through time, etc. These techniques can be very powerful for examining patterns in behavior, especially for psychological institutions who commonly send self-assessment surveys to patients.

# In[32]:


data.head()


# In[34]:


sns.relplot(x='A',y='B',hue='B',data=data)
plt.xticks(rotation=90)
plt.show()


# In[35]:


fix, ax=plt.subplots(2,2,figsize=(12,12))
sns.distplot(data['A'],ax=ax[0,0])
sns.distplot(data['B'],ax=ax[0,1])
sns.distplot(data['B'],ax=ax[1,0])


# distplot plots a univariate distribution of observations. The distplot () function combines the matplotlib hist function with the seaborn kdeplot () and rugplot () functions. The plot below shows a simple distribution. It creats random values with random.randn (). This will work if you manually define values too.

# In[36]:


plt.plot(data.A,data.B)


# In[37]:


axes1 = plt.subplot2grid (
(7, 1), (0, 0), rowspan = 2, colspan = 1)
axes2 = plt.subplot2grid (
(7, 1), (2, 0), rowspan = 2, colspan = 1)
axes1.plot(data.A, data.B)
axes2.plot(data.A, data.B)


# In[38]:


plt.plot(data.A,data.B, color='green', linewidth=3, marker='o', 
         markersize=15, linestyle='--')
plt.title("Line Chart")
plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')
plt.show()


# In[39]:


plt.bar(data.A, data.B, color='green', edgecolor='blue', 
        linewidth=2) 
plt.show()


# In[40]:


def numeric_analysis_hist(feature1):
    sns.set_style('whitegrid')    
    plt.figure(figsize=(15,5))
    plt.title(feature1+' Distribution',fontsize = 20) 
    plt.xlabel(feature1 , fontsize = 15)  
    
    dist = sns.distplot(data[feature1],color='g')


# **After the normalization of data**
# 
# Is the data distribution skewed? If highly skewed, do you still find outliers which you did not treat?¶

# In[41]:


numeric_analysis_hist('A')


# In[42]:


numeric_analysis_hist('B')


# In[43]:


import statsmodels.api as sm
import scipy.stats as norm
import pylab
sm.qqplot(data['A'],line='45')
pylab.show()


# The QQ Plot allows us to see deviation of a normal distribution much better than in a Histogram or box plot.
# 
# Now the distribution of data in this column increases a lot than before and the data too picked at the middle

# In[44]:


plt.hist(data.A, bins=25, color='green', edgecolor='blue',
         linestyle='--', alpha=0.5)
plt.title("A")
plt.ylabel('Frequency')
plt.xlabel('Total')
  
plt.show()


# In[45]:


plt.hist(data.B, bins=25, color='green', edgecolor='blue',
         linestyle='--', alpha=0.5)
plt.title("B")
plt.ylabel('Frequency')
plt.xlabel('Total')
  
plt.show()


# **3D plot**

# In[46]:


sns.set(rc={'figure.figsize': (8, 5)})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.A,data.B, c='r', marker='o')
ax.set_xlabel('X Label'), ax.set_ylabel('Y Label')
plt.show()


# In[47]:


plt.scatter(data.A,data.B)
plt.show()


# In[48]:


plt.hist(data.A)
plt.show()


# Note: It is crucial to have balanced class distribution, i.e., there should be no significant difference between x1 and x2 classes (commonly x1 classes are more than x2 in the life field). The models trained on datasets with balanced class distribution tend to be biased and show good performance toward minor knn.
# 
# The normal distribution is the most important probability distribution in statistics because it fits many natural phenomena. For example, heights, blood pressure, measurement error, and IQ scores follow the normal distribution. It is also known as the Gaussian distribution and the bell curve.

# In[49]:


from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()


# Feature selection for model training For good predictions of the outcome, it is essential to include the good independent variables (features) for fitting the regression model (e.g. variables that are not highly correlated). If you include all features, there are chances that you may not get all significant predictors in the model.

# In[50]:


for col in data.columns:
    if col != 'B':
        print(col)
        data[col] = min_max.fit_transform(data[[col]])


# **Performing Linear Regression**

# In[86]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=0000)


# In[88]:


train_test_split


# In[89]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[90]:


import numpy as np

X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]


# In[91]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[107]:


# import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lr = LinearRegression()

# Fit the model using lr.fit()
lr.fit(X_train, y_train)


# **Coefficients Calculation**

# In[108]:


# Print the intercept and coefficients
print(lr.intercept_)
print(lr.coef_)


# **Predictions**

# In[109]:


# Making predictions on the testing set
y_pred = lr.predict(X_test)


# In[96]:


type(y_pred)


# Computing RMSE and R^2 Values
# RMSE is the standard deviation of the errors which occur when a prediction is made on a dataset. This is the same as MSE (Mean Squared Error) but the root of the value is considered while determining the accuracy of the model

# In[97]:


y_test.shape   # cheek the shape to generate the index for plot


# In[99]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)


# In[100]:


r_squared = r2_score(y_test, y_pred)


# In[101]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[106]:


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred,c='blue')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.grid()


# In[ ]:




