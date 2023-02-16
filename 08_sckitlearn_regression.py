#!/usr/bin/env python
# coding: utf-8

# # Scikit-learn
# 
# Scikit-learn is a machine learning library for Python. It features several regression, classification and clustering algorithms including SVMs, gradient boosting, k-means, random forests and DBSCAN. It is designed to work with Python Numpy and SciPy.
# 
# - **Supervised Learning algorithms** − Almost all the popular supervised learning algorithms, like Linear Regression, Support Vector Machine (SVM), Decision Tree etc., are the part of scikit-learn.
# 
# - **Unsupervised Learning algorithms** − On the other hand, it also has all the popular unsupervised learning algorithms from clustering, factor analysis, PCA (Principal Component Analysis) to unsupervised neural networks.
# 
# - **Clustering** − This model is used for grouping unlabeled data.
# 
# - **Cross Validation** − It is used to check the accuracy of supervised models on unseen data.
# 
# - **Dimensionality Reduction** − It is used for reducing the number of attributes in data which can be further used for summarisation, visualisation and feature selection.
# 
# - **Ensemble methods** − As name suggest, it is used for combining the predictions of multiple supervised models.
# 
# - **Feature extraction** − It is used to extract the features from data to define the attributes in image and text data.
# 
# - **Feature selection** − It is used to identify useful attributes to create supervised models.

# In[28]:


# import sklearn as skl


# ## Regression
# 

# ### Linear Regression assumptions
# 
# Some of the very important assumptions to fitting a regression model includes
# 
# - Independence: Observations are independent of each other.
# - Normality: For any fixed value of X, Y is normally distributed.
# - Linearity: The relationship between X and the mean of Y is linear.

# In[32]:


#libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[33]:


### Examples from Scikit Learn


# In[40]:


## Linear Regression

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[42]:


## Ridge Regression

# Create linear regression object
regr = linear_model.Ridge(alpha=.5)

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# ### Exercise
# 
# Plot the two together

# ### Dataset
# 
# We will use the boston housing dataset which is already preloaded as a scikit learn in-built dataset

# In[6]:


#data
boston = load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
#target variable
boston_df['Price']=boston.target
#preview
boston_df.head()


# In[8]:


#Data dimension
boston_df.shape


# In[10]:


#deescriptives
boston_df.describe()


# In[11]:


#Exploration
plt.figure(figsize = (10, 10))
sns.heatmap(boston_df.corr(), annot = True)


# ### Linear Regression assumptions
# 
# Some of the very important assumptions to fitting a regression model includes
# 
# - Independence: Observations are independent of each other.
# - Normality: For any fixed value of X, Y is normally distributed.
# - Linearity: The relationship between X and the mean of Y is linear.
# 
# However, we see strong correlation between features (x) which is termed multicolinearity. We will need to drop some columns leading to these strong correlations values. Y

# In[12]:


#There are cases of multicolinearity, we will drop a few columns
boston_df.drop(columns = ["INDUS", "NOX"], inplace = True)


# In[14]:


boston_df.head()


# In[15]:


#pairplot
sns.pairplot(boston_df)


# Variables should be normally distributed and linear. However, the relationship between LSTAT and Price is nonlinear. Hence, we log it.

# In[17]:


#we will log the LSTAT Column
boston_df.LSTAT = np.log(boston_df.LSTAT)


# In[19]:


#pairplot again
#pairplot
sns.pairplot(boston_df)


# ### Data split and scaling

# In[20]:


#preview
features = boston_df.columns[0:11]
target = boston_df.columns[-1]

#X and y values
X = boston_df[features].values
y = boston_df[target].values

#splot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

print("The dimension of X_train is {}".format(X_train.shape))
print("The dimension of X_test is {}".format(X_test.shape))
#Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[22]:


#Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Model fitting
# 
# 
# - Fit linear regression modeL and score both train and test set
# - Fit a Score regression model and score both train and test set
# 
# Compare results

# In[36]:


#Model
lr = LinearRegression()

#Fit model
lr.fit(X_train, y_train)

#predict
y_pred = lr.predict(X_test)

#actual
actual = y_test

train_score_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)

print("The train score for lr model is {}".format(train_score_lr))
print("The test score for lr model is {}".format(test_score_lr))


# ### Exercise
# 
# - Plot the test and predicted data
# - How well does linear regression performs?

# In[26]:



#Ridge Regression Model
ridgeReg = Ridge(alpha=10)

ridgeReg.fit(X_train,y_train)

#train and test scorefor ridge regression
train_score_ridge = ridgeReg.score(X_train, y_train)
test_score_ridge = ridgeReg.score(X_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))


# ### Exercise
# 
# - Plot the test and predicted data
# - How well does ridge regression performs?

# ### References 
# - https://www.tutorialspoint.com/scikit_learn/scikit_learn_clustering_methods.htm
# 
