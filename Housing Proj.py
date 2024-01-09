#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from js import fetch
import io

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
resp = await fetch(url)
boston_url = io.BytesIO((await resp.arrayBuffer()).to_py())
boston_df=pd.read_csv(boston_url)

boston_df.head()

#The following describes the dataset variables:
#
#      CRIM - per capita crime rate by town
#      ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
#      INDUS - proportion of non-retail business acres per town.
#      CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
#      NOX - nitric oxides concentration (parts per 10 million)
#      RM - average number of rooms per dwelling
#      AGE - proportion of owner-occupied units built prior to 1940
#      DIS - weighted distances to five Boston employment centres
#      RAD - index of accessibility to radial highways
#      TAX - full-value property-tax rate per $10,000
#      PTRATIO - pupil-teacher ratio by town
#      LSTAT - % lower status of the population
#      MEDV - Median value of owner-occupied homes in $1000's


# In[67]:


#TASK 4
#    For the "Median value of owner-occupied homes" provide a boxplot
ax = sns.boxplot(y='MEDV', data=boston_df)
pyplot.ylabel('MEDV')
pyplot.title('Median value of owner-occupied homes')
pyplot.show()

#    Provide a  bar plot for the Charles river variable
ax = sns.barplot(y='CHAS', data=boston_df)
pyplot.ylabel('CHAS')
pyplot.title('Charles river variable')
pyplot.show()

#    Provide a boxplot for the MEDV variable vs the AGE variable. (Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)
boston_df.loc[(boston_df['AGE'] <= 35), 'AGE_group'] = '35 years and younger'
boston_df.loc[(boston_df['AGE'] > 35)&(boston_df['AGE'] < 70), 'AGE_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df['AGE'] >= 70), 'AGE_group'] = '70 years and older'
ax = sns.boxplot(x='AGE_group', y='MEDV', data=boston_df)
pyplot.xlabel('AGE Group')
pyplot.ylabel('MEDV')
pyplot.title('Median value of owner-occupied homes')
pyplot.show()

#    Provide a scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town. What can you say about the relationship?
ax = sns.scatterplot(x='INDUS', y='NOX', data=boston_df)
pyplot.xlabel('INDUS')
pyplot.ylabel('NOX')
pyplot.title('Nitric Oxides Concentration')
pyplot.show()

#    Create a histogram for the pupil to teacher ratio variable
sns.catplot(x='PTRATIO', kind='count', data=boston_df)
pyplot.xlabel('PTRATIO')
pyplot.title('Pupil to Teacher Ratio')
pyplot.show()

#boston_df['PTRATIO'].min()
#boston_df['PTRATIO'].max()


# In[68]:


#TASK 5
#For each of the following questions;

####    Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)

####   State the Hypothesis:
####   $H_0: µ1 = µ2$("There is no difference in median value of houses bounded by the Charles river")
####   $H_1: µ1 ≠ µ2$("There is a difference in median value of houses bounded by the Charles river")

scipy.stats.levene(boston_df[boston_df['CHAS'] == 1]['MEDV'],
                   boston_df[boston_df['CHAS'] == 0]['MEDV'], center='mean')
#### Conclusion: Since the p-value is less than 0.05, the variance are not equal, for the purposes of this exercise, we will move along
# LeveneResult(statistic=8.75190489604598, pvalue=0.003238119367639829)

# In[69]:

scipy.stats.ttest_ind(boston_df[boston_df['CHAS'] == 1]['MEDV'],
                   boston_df[boston_df['CHAS'] == 0]['MEDV'], equal_var = True)
#### Conclusion: Since the p-value is less than alpha value 0.05, we reject the null hypothesis as there is proof that there is a statistical difference
# Ttest_indResult(statistic=3.996437466090509, pvalue=7.390623170519905e-05)


# In[72]:

####    Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)

####   State the Hypothesis:
####   $H_0: µ1 = µ2$("There is no difference in median value of houses for each proportion of owner occupied units")
####   $H_1: µ1 ≠ µ2$("There is a difference in median value of houses for each proportion of owner occupied units")

boston_df.loc[(boston_df['AGE'] <= 35), 'AGE_group'] = '35 years and younger'
boston_df.loc[(boston_df['AGE'] > 35)&(boston_df['AGE'] < 70), 'AGE_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df['AGE'] >= 70), 'AGE_group'] = '70 years and older'

scipy.stats.levene(boston_df[boston_df['AGE_group'] == '35 years and younger']['MEDV'],
                   boston_df[boston_df['AGE_group'] == 'between 35 and 70 years']['MEDV'], 
                   boston_df[boston_df['AGE_group'] == '70 years and older']['MEDV'], 
                   center='mean')
#### Conclusion: Since the p-value is greater than 0.05, the variance are equal
# LeveneResult(statistic=2.7806200293748304, pvalue=0.06295337343259205)


# In[73]:

thirtyfive_lower = boston_df[boston_df['AGE_group'] == '35 years and younger']['MEDV']
thirtyfive_seventy = boston_df[boston_df['AGE_group'] == 'between 35 and 70 years']['MEDV']
seventy_older = boston_df[boston_df['AGE_group'] == '70 years and older']['MEDV']

f_statistic, p_value = scipy.stats.f_oneway(thirtyfive_lower, thirtyfive_seventy, seventy_older)
print("F_Statistic: {0}, P-Value: {1}".format(f_statistic,p_value))
#### Conclusion: Since the p-value is less than 0.05, we will reject the null hypothesis as there is significant evidence that at least one of the means differ
# F_Statistic: 36.40764999196599, P-Value: 1.7105011022702984e-15



# In[74]:

####    Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)

####   State the Hypothesis:
####   $H_0: µ1 = µ2$("There is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town")
####   $H_1: µ1 ≠ µ2$("There is a relationship between Nitric oxide concentrations and proportion of non-retail business acres per town")

ax = sns.scatterplot(x="INDUS", y="NOX", data=boston_df)
pyplot.xlabel('INDUS')
pyplot.ylabel('NOX')
pyplot.title('Nitric Oxides Concentration')
pyplot.show()

# In[86]:

scipy.stats.pearsonr(boston_df['INDUS'], boston_df['NOX'])

#### Conclusion: Since the p-value (Sig. (2-tailed) < 0.05, we reject the Null hypothesis and conclude that there exists a relationship between proportion of non-retail business acres per town and Nitric oxide concentration
# PearsonRResult(statistic=0.7636514469209151, pvalue=7.913361061239527e-98


# In[85]:


####    What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)

####   State the Hypothesis:
####   $H_0: µ1 = µ2$("Weighted distance to the five Boston employment centres has no impact on the median value of owner occupied homes")
####   $H_1: µ1 ≠ µ2$("Weighted distance to the five Boston employment centres has an impact on the median value of owner occupied homes")

Weighted distance to the five Boston employment centres has no impact on the median value of owner occupied homes

## X is the input variables (or independent variables)
X = boston_df['DIS']
## y is the target/dependent variable
y = boston_df['MEDV']
## add an intercept (beta_0) to our model
X = sm.add_constant(X) 

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()

#### Conclusion: There is no correlation between the weighted distances to the five Boston centres and median value of owner occupied homes
#### Impact of weighted distance is median value of owner occupied homes is positive (coeff = 1.0916)

