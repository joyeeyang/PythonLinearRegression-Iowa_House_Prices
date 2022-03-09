#!/usr/bin/env python
# coding: utf-8

# In[1]:


1# Importing necessary packages
import pandas as pd # python's data handling spackage
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression


# ##### Loading data.

# In[2]:


# Both features and target have already been scaled: mean = 0; SD = 1

data = pd.read_csv("Houseprice_data_scaled.csv")


# In[3]:


# First 1800 data items are training set; the next 600 are the validation set
train = data.iloc[:1800] 
val = data.iloc[1800:2400]


# In[4]:


# Creating the "X" and "y" variables. We drop sale price from "X"
X_train, X_val = train.drop('Sale Price', axis=1), val.drop('Sale Price', axis=1)
y_train, y_val = train[['Sale Price']], val[['Sale Price']] 


# ###### Linear Regression

# In[5]:


# Importing models
from sklearn.linear_model import LinearRegression


# In[6]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[7]:


# Create dataFrame with corresponding feature and its respective coefficients
coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lr.intercept_) + list(lr.coef_[0])
    ]
).transpose().set_index(0)
coeffs


# In[8]:


# calculate the mse of the regression
pred = lr.predict(X_val)
mse_original = mse(y_val, pred)
print(mse_original)


# ###### Ridge Regression

# In[9]:


# Importing Ridge
from sklearn.linear_model import Ridge


# In[10]:


# The alpha used by Python's ridge should be the lambda in Hull's book times the number of observations
alphas=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.4*1800]
mses=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_train,y_train)
    pred=ridge.predict(X_val)
    mses.append(mse(y_val,pred))
    print(mse(y_val,pred))


# In[11]:


plt.plot(alphas, mses)


# ###### Lasso

# In[12]:


# Import Lasso
from sklearn.linear_model import Lasso


# In[13]:


# Here we produce results for alpha=0.05 which corresponds to lambda=0.1 in Hull's book
lasso = Lasso(alpha=0.05)
lasso.fit(X_train, y_train)


# In[14]:


# DataFrame with corresponding feature and its respective coefficients
coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lasso.intercept_) + list(lasso.coef_)
    ]
).transpose().set_index(0)
coeffs


# ###### Lasso with different levels of alpha and its mse

# In[15]:


# We now consider different lambda values. The alphas are half the lambdas
alphas=[0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train,y_train)
    pred=lasso.predict(X_val)
    mses.append(mse(y_val,pred))
    print(mse(y_val, pred))


# In[16]:


plt.plot(alphas, mses)


# ###  Data with Additional Features (Cleaning Method #1)

# In[17]:


# 4 additional features are added to the data: 
# LotFrontage, LotShape which is broken down into 3 dummy variables: IR1, IR2, IR3, Yr Sold, SaleCondition (5 dummies)
url = 'https://drive.google.com/file/d/160U3Nw_KwWM2gz5bzFdBi96KIHklPWKH/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
data_add = pd.read_csv(path)

data_add.head(10)


# In[18]:


#Data cleaning for LotFrontage
#First method: replace missing value with median

frontage_median = data_add['LotFrontage'].median()
data_add['LotFrontage'] = data_add['LotFrontage'].fillna(frontage_median)

#Scale LotFrontage, IR1, IR2, IR3, YrSold, and the 5 dummies for SaleConditon: Abnormal, AdjLand, Alloca, Family, Partial

for col in ['LotFrontage', 'IR1', 'IR2', 'IR3', 'YrSold', 'Abnorml', 'AdjLand', 'Alloca', 'Family','Partial']:
    data_add[col] = (data_add[col] - data_add[col].mean())/data_add[col].std()

data_add.head(10)


# In[19]:


# Split the data into training set and validation set
train_add = data_add.iloc[:1800] 
val_add = data_add.iloc[1800:2400]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train_add, X_val_add = train_add.drop('Sale Price', axis=1), val_add.drop('Sale Price', axis=1)
y_train_add, y_val_add = train_add[['Sale Price']], val_add[['Sale Price']] 

print(y_train_add.head())


# ### Linear Regression Additional Features (for Cleaning Method #1)

# In[20]:


lr=LinearRegression()
lr.fit(X_train_add,y_train_add)


# In[21]:


# Create dataFrame with corresponding feature and its respective coefficients
coeffs_add = pd.DataFrame(
    [
        ['intercept'] + list(X_train_add.columns),
        list(lr.intercept_) + list(lr.coef_[0])
    ]
).transpose().set_index(0)
coeffs_add

pred = lr.predict(X_train_add)
mse_add1 = mse(y_train_add, pred)
print(mse_add1)


# In[22]:


# Compare the two regression models
mse_add = mse(y_val_add, lr.predict(X_val_add))


print("MSE of original regression: " + str(mse_original))
print("MSE of training set with additional features: " + str(mse_add1))
print("MSE of validation set regression with additional features: " + str(mse_add))


# In[23]:


#Ridge Regression with Additional features

alphas=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.4*1800]
mses=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_train_add,y_train_add)
    pred=ridge.predict(X_val_add)
    mses.append(mse(y_val_add,pred))
    print(mse(y_val_add,pred))


# In[24]:


#Using a = 0.01*1800 for ridge regression
ridge = Ridge(alpha=0.01*1800)
ridge.fit(X_train_add, y_train_add)
list1 = ['intercept'] + list(X_train_add.columns)
list2 = list(ridge.intercept_)
list3 = ridge.coef_[0]
list2.extend(list3)

coeffs = pd.DataFrame(
    {
        'feature': list1,
        'coefficient': list2
    }
)
coeffs


# In[25]:


#Using a = 0.05 first lasso regression
lasso = Lasso(alpha=0.05)
lasso.fit(X_train_add, y_train_add)
coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train_add.columns),
        list(lasso.intercept_) + list(lasso.coef_)
    ]
).transpose().set_index(0)
coeffs


# In[26]:


#Lasso Regression with Additional features
alphas_add=[0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas_add:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_train_add,y_train_add)
    pred=lasso.predict(X_val_add)
    mses.append(mse(y_val_add,pred))
    print(mse(y_val_add, pred))


# In[27]:


#Randomly splitting data into training, validation, and test set

from sklearn.model_selection import train_test_split


train_size = 1800/2908
valid_size = 600/2908
test_size = 508/2908


X = X_train_add
y = y_train_add

X_rtrain, X_rtest, y_rtrain, y_rtest = train_test_split(X, y, test_size=1 - train_size)
X_rvalid, X_rtest, y_rvalid, y_rtest = train_test_split(X_rtest,y_rtest, test_size=test_size/(test_size + valid_size))

print(X_rtrain, X_rtest, y_rtrain, y_rtest)
print(X_rvalid, X_rtest, y_rvalid, y_rtest)


# In[28]:


lr=LinearRegression()
lr.fit(X_rtrain,y_rtrain)

#Create dataFrame with corresponding feature and its respective coefficients
coeffs_add = pd.DataFrame(
    [
        ['intercept'] + list(X_rtrain.columns),
        list(lr.intercept_) + list(lr.coef_[0])
    ]
).transpose().set_index(0)
coeffs_add


# In[29]:


# Compare the three regression models
pred1 = lr.predict(X_rtrain)
mse_rand_train = mse(y_rtrain, pred1)
print(mse_rand_train)

pred = lr.predict(X_rvalid)
mse_add_rand = mse(y_rvalid, pred)

print("MSE of original regression:" + str(mse_original))
print("MSE of regression with additional features:" + str(mse_add))
print("MSE of training set regression with random split and additional features:" + str(mse_rand_train))
print("MSE of validation set regression with random split and additional features:" + str(mse_add_rand))


# In[30]:


#Ridge regression alpha = 0.1*1800, weights

#Using a = 0.01*1800 for ridge regression
ridge_r = Ridge(alpha=0.01*1800)
ridge_r.fit(X_rtrain, y_rtrain)
listr1 = ['intercept'] + list(X_rtrain.columns)
listr2 = list(ridge_r.intercept_)
listr3 = ridge_r.coef_[0]
listr2.extend(listr3)

coeffs_r = pd.DataFrame(
    {
        'feature': listr1,
        'coefficient': listr2
    }
)
coeffs_r


# In[31]:


#Ridge Regression with Random Split different alphas

alphas_add=[0.01*1800, 0.02*1800, 0.03*1800, 0.04*1800, 0.05*1800, 0.075*1800,0.1*1800,0.2*1800, 0.4*1800]
mses=[]
for alpha in alphas_add:
    ridge=Ridge(alpha=alpha)
    ridge.fit(X_rtrain,y_rtrain)
    pred=ridge.predict(X_rvalid)
    mses.append(mse(y_rvalid,pred))
    print(mse(y_rvalid,pred))


# In[32]:


#Using a = 0.05 first lasso regression random split
lasso = Lasso(alpha=0.05)
lasso.fit(X_rtrain, y_rtrain)
coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_rtrain.columns),
        list(lasso.intercept_) + list(lasso.coef_)
    ]
).transpose().set_index(0)
coeffs


# In[33]:


#Lasso Regression with Random Split
alphas=[0.01/2, 0.02/2, 0.03/2, 0.04/2, 0.05/2, 0.075/2, 0.1/2]
mses=[]
for alpha in alphas:
    lasso=Lasso(alpha=alpha)
    lasso.fit(X_rtrain,y_rtrain)
    pred=lasso.predict(X_rvalid)
    mses.append(mse(y_rvalid,pred))
    print(mse(y_rvalid, pred))


# ## Data with Additional Features (Cleaning Method #2)
# 

# In[34]:


# 4 additional features are added to the data: 
# LotFrontage, LotShape which is broken down into 3 dummy variables: IR1, IR2, IR3, Yr Sold, SaleCondition (5 dummies)
data_add2 = pd.read_csv("Houseprice_data_additional_features.csv")

data_add2.head(10)


# In[35]:


#Data cleaning for LotFrontage
#Second method: replace missing value with mean 

frontage_mean = data_add2['LotFrontage'].mean()
data_add2['LotFrontage'] = data_add2['LotFrontage'].fillna(frontage_mean)


#Scale LotFrontage, IR1, IR2, IR3, YrSold, and the 5 dummies for SaleCondition: Abnormal, AdjLand, Alloca, Family, Partial

for col in ['LotFrontage', 'IR1', 'IR2', 'IR3', 'YrSold', 'Abnorml', 'AdjLand', 'Alloca', 'Family','Partial']:
    data_add2[col] = (data_add2[col] - data_add2[col].mean())/data_add2[col].std()

data_add2.head(10)


# In[36]:


# Split the data into training set and validation set
train_add2 = data_add2.iloc[:1800] 
val_add2 = data_add2.iloc[1800:2400]

# Creating the "X" and "y" variables. We drop sale price from "X"
X_train_add2, X_val_add2 = train_add2.drop('Sale Price', axis=1), val_add2.drop('Sale Price', axis=1)
y_train_add2, y_val_add2 = train_add2[['Sale Price']], val_add2[['Sale Price']]


# ## Linear Regression Additional Features (for Cleaning Method #2)
# 

# In[37]:


lr=LinearRegression()
lr.fit(X_train_add2,y_train_add2)
# Create dataFrame with corresponding feature and its respective coefficients
coeffs_add2 = pd.DataFrame(
    [
        ['intercept'] + list(X_train_add2.columns),
        list(lr.intercept_) + list(lr.coef_[0])
    ]
).transpose().set_index(0)

coeffs_add2


# In[38]:


# Compare regression model with Method #1 (3) regression models 

pred_2 = lr.predict(X_train_add2)
mse_train_2 = mse(y_train_add2, pred_2)
pred_3 = lr.predict(X_val_add2)
mse_add2 = mse(y_val_add2, pred_3)

print("MSE of original regression:" + str(mse_original))
print("MSE of training set regression with additional features (Method #2):" + str(mse_train_2))
print("MSE of validation set regression with additional features (Method #1):" + str(mse_add))
print("MSE of validation set regression with additional features (Method #2):" + str(mse_add2))
print("MSE of training set regression with random split and additional features:" + str(mse_rand_train))
print("MSE of validation set regression with random split and additional features:" + str(mse_add_rand))


# In[ ]:




