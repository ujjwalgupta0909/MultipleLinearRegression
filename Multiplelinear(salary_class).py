# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:46:38 2018

@author: hp
"""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset
dataset=pd.read_csv("Salary_Classification.csv")
features=dataset.iloc[:,0:-1].values
labels=dataset.iloc[:,-1].values

#taking care of categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
features[:,0]=labelencoder.fit_transform(features[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
features=onehotencoder.fit_transform(features).toarray()

#dummy variable trap
features=features[:,1:]

#splitting the dataset
from sklearn.cross_validation import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)

#creating a model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(features_train,labels_train)

#predicting the values
pred=regressor.predict(features_test)

#score
score=regressor.score(features_train,labels_train)

#BUILDING AN OPTIMAL MODEL (BACKWARD ELIMINATION)
import statsmodels.formula.api as sm
features=np.append(arr=np.ones((30,1)).astype(int),values=features,axis=1)
features_opt=features[:,[0,1,2,3,4,5]]
regressor_ols= sm.OLS(endog=labels,exog=features_opt).fit()
regressor_ols.summary()

features_opt=features[:,[0,1,3,4,5]]
regressor_ols= sm.OLS(endog=labels,exog=features_opt).fit()
regressor_ols.summary()

features_opt=features[:,[0,1,3,5]]
regressor_ols= sm.OLS(endog=labels,exog=features_opt).fit()
regressor_ols.summary()

features_opt=features[:,[0,3,5]]
regressor_ols= sm.OLS(endog=labels,exog=features_opt).fit()
regressor_ols.summary()

features_opt=features[:,[0,5]]
regressor_ols= sm.OLS(endog=labels,exog=features_opt).fit()
regressor_ols.summary()
