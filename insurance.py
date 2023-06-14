# -*- coding: utf-8 -*-
"""
**********Linear Regression**********
**********@author: gulsen************
"""
""" *********************************************************************** """
""" 
Libraries
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy as sp
plt.rcParams['figure.figsize']=[8,5]
plt.rcParams['font.size']=14
plt.rcParams['font.weight']='bold'
plt.style.use('seaborn-whitegrid')

""" *********************************************************************** """
"""
Data import
"""

path = r"C:\Users\LENOVO\Desktop\data_science\Insurance_Data\insurance.csv"

data1 = pd.read_csv(path)

data1.shape #m=1338 training example, n=7 independent samples (1338,7)

""" *********************************************************************** """

"""
VISUALIZE
"""

sns.lmplot(x='bmi', y='charges', data=data1, aspect=2,height=6) #aspect: aspect ratio of each facet
plt.xlabel('Body Mass Index$(kg/m^2$): as Independent Variable')
plt.ylabel('Insurance Charges: as Dependent Variable')
plt.title('Charge Vs BMI');

#data1.describe()

plt.figure(figsize=(12,4)) #for missing value in the dataset
sns.heatmap(data1.isnull(), cbar=False, cmap='viridis', yticklabels=False) #cmap: list of colors
plt.title('Missing value in dataset');

corr=data1.corr()
sns.heatmap(corr, cmap='Wistia', annot=True) #annot: if True, write the data value in each cell.
# There no correlation among variables.

#Histogram:
f=plt.figure(figsize=(12,4))
ax=f.add_subplot(111) # 111:size 
sns.distplot(data1['charges'],bins=50, color='r', ax=ax) #bins:the number of bars
ax.set_title('Distribution of insurance charges')

f=plt.figure(figsize=(12,4))
ax=f.add_subplot(111) 
sns.distplot(data1['charges'],bins=40, color='b', ax=ax) #bins:the number of bars
ax.set_title('Distribution of insurance charges in $log$ scale')
ax.set_xscale('log');
#Charges varies from 1120 to 63500, the plot is right skewed.

#Violin:
f=plt.figure(figsize=(14,6))
ax=f.add_subplot(121)
sns.violinplot(x='sex', y='charges', data=data1, palette='Wistia', ax=ax)
ax.set_title('Violin plot of Charges vs sex')

ax=f.add_subplot(122)
sns.violinplot(x='smoker', y='charges', data=data1, palette='magma', ax=ax)
ax.set_title('Violin plot of Charges vs smoker')

ax=f.add_subplot(122)
sns.violinplot(x='sex',y='bmi', data=data1, palette='Blues')
ax.set_title('Violin plot of Sex vs bmi')

ax=f.add_subplot(122)
sns.violinplot(x='age',y='smoker', data=data1, palette='Pastel1')
ax.set_title('Violin plot of Age vs smoker')

#Box Plot:
plt.figure(figsize=(14,6))
sns.boxplot(x='children', y='charges',hue='sex', data=data1, palette='Pastel2')
plt.title('Box plot of charges of children')

data1.groupby('children').agg(['mean','min','max'])['charges']

plt.figure(figsize=(14,6))
sns.violinplot(x='region', y='charges', hue='sex', data=data1, palette='rainbow')
plt.title('Violin plot of charges vs children');

#Scatter Plot:
f=plt.figure(figsize=(14,6))
ax=f.add_subplot(121)
sns.scatterplot(x='age',y='charges',data=data1, palette='pink',hue='smoker', ax=ax)
ax.set_title('Scatter plot of Charges vs age')


ax=f.add_subplot(122)
sns.scatterplot(x='bmi',y='charges',data=data1, palette='viridis',hue='smoker', ax=ax)
ax.set_title('Scatter plot of Charges vs bmi')
plt.savefig('sc.png'); # save the current figure.

""" *********************************************************************** """

"""
Data PROCESSİNG
"""

#Dummy variable

categorical_columns=['sex','children','smoker','region']
data1_encode=pd.get_dummies(data=data1, prefix='OHE', prefix_sep='_', #OHE: One-Hot Encoding 
                            columns=categorical_columns,
                            drop_first=True,
                            dtype='int8')

print('Columns in original data frame:\n', data1.columns.values)
print('\nNumber of rows and columns in the dataset:', data1.shape)
print('\nColumns in data frame after encoding dummy variable:\n', data1_encode.columns.values)
print('\nNumber of rows and columns in the dataset:', data1_encode.shape)

#Box-Cox transformation for normality
y_bc, lam, ci=boxcox(data1_encode['charges'], alpha=0.05)
ci,lam

data1_encode['charges']=np.log(data1_encode['charges'])

#Train Test Split

X=data1_encode.drop('charges', axis=1) #Independent variable
y=data1_encode['charges'] #Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=23)

#Model Building

X_train_0=np.c_[np.ones((X_train.shape[0],1)),X_train] #add x0 =1 to dataset
X_test_0=np.c_[np.ones((X_train.shape[0],1)),X_test] 

theta=np.matmul(np.linalg.inv(np.matmul(X_train_0.T,X_train_0)), np.matmul(X_train_0.T,y_train)) #build model : θ=(X^TX)−1X^Ty 

parameter=['theta_'+str(i) for i in range (X_train_0.shape[1])] #The parameters for linear regression model
columns=['instersect:x_0=1']+list(X.columns.values)
parameter_data1=pd.DataFrame({'Parameter': parameter,'Columns':columns,'theta':theta})

lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

sk_theta=[lin_reg.intercept_]+list(lin_reg.coef_)
parameter_data1=parameter_data1.join(pd.Series(sk_theta, name='Sklearn_theta'))
parameter_data1

#Model Evaluation

y_pred_norm = np.matmul(X_test_0,theta) #Normal equation

J_mse= np.sum((y_pred_norm-y_test)**2) / X_test_0.shape[0] #Evaluvation: MSE

#R_square:
    
sse=np.sum((y_pred_norm - y_test)**2) #Sum of square error
sst=np.sum((y_test - y_test.mean())**2) #Sum of square total
R_square=1-(sse/sst)
print('The Mean Square Error(MSE) or J(theta) is:' , J_mse)
print('R square obtain for normal equation method is:', R_square)

#sklearn regression module:
y_pred_sk=lin_reg.predict(X_test)

#Evaluvation: MSE

J_mse_sk=mean_squared_error(y_pred_sk, y_test)

#R_square
R_square_sk = lin_reg.score(X_test,y_test)
print('The Mean Square Error(MSE) or J(theta) is: ', J_mse_sk)
print('R square obtain for sckit learn library is:', R_square_sk)

#****The model returns  R2 value of 77.95%, we have transformer out variable by applying natural log.****

#Model Validation

f = plt.figure(figsize=(14,5))
ax=f.add_subplot(133)
sns.scatterplot(x=y_test, y=y_pred_sk, ax=ax, color='r')
ax.set_title('Check for Linearity:\n Actual vs Predicted value')

ax=f.add_subplot(122)
sns.distplot((y_test - y_pred_sk), ax=ax, color='b')
ax.axvline((y_test - y_pred_sk).mean(), color='k', linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual error')

f,ax=plt.subplots(1,2,figsize=(14,6))
_,(_,_,r)=sp.stats.probplot((y_test - y_pred_sk), fit=True, plot=ax[0])
ax[0].set_title('check for Multivariate Normality: \nQ-Q Plot')

sns.scatterplot(y=(y_test - y_pred_sk),x=y_pred_sk, ax=ax[1], color='r')
ax[1].set_title('Check for Homoscedasticity: \nResidual Vs Predicted')

VIF= 1/(1-R_square_sk) # Check for Multicollinearity: If VIF >1 & VIF <5 moderate correlation
VIF

