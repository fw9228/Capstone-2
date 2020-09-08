#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# **Importing Libraries**

# In[1]:


import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split

import os


# **Importing data**

# The Lending Club Dataset is used for this project: a large US peer-to-peer lending company. There are several different versions of this dataset. We have used the updated dataset (version 2), which is available on kaggle: https://www.kaggle.com/wendykan/lending-club-loan-data/
# 
# We divided the data into two periods because we assume that some data are available at the moment when we need to build Expected Loss models, and some data comes from applications after. Later, we investigate whether the applications we have after we built the Probability of Default (PD) model have similar characteristics with the applications we used to build the PD model.

# In[2]:


loan = r'F:\Data Analysis\Springboard\Data Science Career Track\Projects\Capstone 2\lending club loan data_version 2\loan.csv'

loan_data_backup = pd.read_csv(loan)

loan_data = loan_data_backup.copy()


# **Explore Data**

# In[3]:


loan_data.head()


# In[4]:


loan_data.tail()


# In[5]:


#Display all columns
#pd.options.display.max_columns = None
#loan_data


# In[6]:


#Display all rows
#pd.options.display.max_rows = None
#loan_data


# In[7]:


loan_data.columns.values


# In[8]:


# Displays column names, complete (non-missing) cases per column, and datatype per column.
loan_data.info()


# In[9]:


loan_data.dtypes


# In[10]:


loan_data.shape


# # DATA PREPROCESSING

# **Pre-processing few continuous variables: `emp_length`, `earlist_cr_line`, `term`, `issue_d`**

# In[11]:


# Display unique values of a column.
loan_data['emp_length'].unique()


# The `emp_length` has four things we have to remove to be able to convert it into an integer:
#     1.`+ years`
#     2.`< 1 year`
#     3.`nan`
#     4.` years` and `year`  (space years and space year)

# In[12]:


#Coverting the employment lenght from object into integer. We will store the new variable as 'employment length int'

# 1.Assign the new ‘employment length int’ to be equal to the ‘employment length’ variable with the string ‘+ years’
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('\+ years', '')

# 2Replace the whole string ‘less than 1 year’ with the string ‘0’.
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))

# 3.Replace the ‘n/a’ string with the string ‘0’.
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',  str(0))

# 4.Replace the string ‘space years’  and 'space year' with nothing.
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')


# In[13]:


# Checks the datatype of a single element of a column.
type(loan_data['emp_length_int'][0])


# Now we transform it into numeric

# In[14]:


loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])


# In[15]:


type(loan_data['emp_length_int'][0])


# Next is `earliest credit line`

# In[16]:


# Next is 'earliest credit line'
#loan_data['earliest_cr_line']


# In[17]:


type(loan_data['earliest_cr_line'][0])


# In[18]:


# 'earliest credit line' is a date variable. We can extracts the date and the time from a string variable that is in a given format.
#loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format= '%b-%Y')

#loan_data['earliest_cr_line_date'] = loan_data['earliest_cr_line'].apply(pd.to_datetime)

loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], infer_datetime_format=True)


# In[19]:


type(loan_data['earliest_cr_line'][0])


# In[20]:


# Calculates the difference between two dates and times.
pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']


# In[21]:


# Assume we are now in December 2017.We calculate the difference between two dates in months, turn it to numeric datatype and round it.
# We save the result in a new variable.
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))


# In[22]:


# Shows some descriptive statisics for the values of a column.
# Dates from 1969 and before are not being converted well, i.e., they have become 2069 and similar,
# and negative differences are being calculated.
loan_data['mths_since_earliest_cr_line'].describe()


# In[23]:


###
# We take three columns from the dataframe. Then, we display them only for the rows where a variable has negative value.
loan_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]


# In[24]:


###
# We set the rows that had negative differences to the maximum value.
loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['mths_since_earliest_cr_line'].max()


# In[25]:


# Calculates and shows the minimum value of a column.
min(loan_data['mths_since_earliest_cr_line'])


# Next variable to pre-process is `term`

# In[26]:


loan_data['term']


# In[27]:


# Shows some descriptive statisics for the values of a column.
loan_data['term'].describe()


# In[28]:


# We replace a string with another string, in this case, with an empty strng (i.e. with nothing).
loan_data['term_int'] = loan_data['term'].str.replace(' months', '')


# In[29]:


type(loan_data['term_int'][0])


# In[30]:


# We remplace a string from a variable with another string, in this case, with an empty strng (i.e. with nothing).
# We turn the result to numeric datatype and save it in another variable.
loan_data['term_int'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))


# In[31]:


# Checks the datatype of a single element of a column.
type(loan_data['term_int'][0])


# Next variable to pre-process is `issue_d`

# In[32]:


loan_data['issue_d']


# In[33]:


# Assume we are now in December 2017
# Extracts the date and the time from a string variable that is in a given format.
loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], infer_datetime_format=True)

# We calculate the difference between two dates in months, turn it to numeric datatype and round it.
# We save the result in a new variable.
loan_data['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['issue_d_date']) / np.timedelta64(1, 'M')))

# Shows some descriptive statisics for the values of a column.
loan_data['mths_since_issue_d'].describe()


# **Preprocessing few discrete/categorical variables**
# 
# **Variables: `grade`, `sub_grade`, `home_owenership`, `purpose`, `addr_state`, `initial_list_status`. We are not going to use `sub_grade`, as it overlaps with grade.**

# In[34]:


# Displays column names, complete (non-missing) cases per column, and datatype per column.
loan_data.info()


# In[35]:


loan_data.dtypes


# In[36]:


# We create dummy variables from all 8 original independent variables, and save them into a list.

loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':'),
                     pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep = ':'),
                     pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep = ':'),
                     pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':')]


# In[37]:


# We concatenate the dummy variables and this turns them into a dataframe.
loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)


# In[38]:


type(loan_data_dummies)


# In[39]:


# We concatenate the original loan_data with the dataframe with dummy variables, along the columns. 
loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)


# In[40]:


# Displays all column names.
loan_data.columns.values


# # DATA CLEANING
# 
# **Checking and handling missing and NA values**
# 
# We will be using these variables in our analysis. Let's check for their missing values and fill them: `annual_inc`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `total_acc`, `acc_now_delinq`, `total_rev_hi_lim`, `emp_length_int`, and `mths_since_earliest_cr_line`

# In[41]:


# It returns 'False' if a value is not missing and 'True' if a value is missing, for each value in a dataframe.
loan_data.isnull()


# In[42]:


# Sets the pandas dataframe options to display all columns/ rows.
#pd.options.display.max_rows = None
#loan_data.isnull().sum()


# In[43]:


# Sets the pandas dataframe options to display all columns/ rows.
#pd.options.display.max_columns = None
#loan_data.isnull().sum()


# **One way to deal with missing values is to remove all observations(rows) where we have missing values. Another way to deal with missing values is to impute them**

# Let's start with `total_rev_hi_lim`

# In[44]:


#'Total revolving high credit/ credit limit' will most likely be equal to 'funded_amnt'.
#So we replace the missing values in that with the values from funded_amnt
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)


# In[45]:


#Checking to see if there are still any missing values
loan_data['total_rev_hi_lim'].isnull().sum()


# Let's do for `annual_inc`

# In[46]:


# We will fill the missing values with the mean value of the non-missing values for 'annual_inc'.
loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)


# In[47]:


loan_data['annual_inc'].isnull().sum()


# For the others, we will fill them with zeros

# In[48]:


# We fill the missing values with zeroes.
loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)


# # Probability of Default (PD) model
# 
# **Data Preparations**

# The Dependent Variable will be Good/ Bad (Default) loan. The definition used here is that, accounts are considered as default (bad loan) if the borrower has been 90 days past due on the loan. Also, a borrower is considered default if the borrower commits fraud. The variable `loan_status` is used to determine if a customer has defaulted or not.

# In[49]:


# Displays unique values of loan_status column
loan_data['loan_status'].unique()


# In[50]:


# Calculates the number of observations for each unique value of a variable
loan_data['loan_status'].value_counts()


# In[51]:


#Total loans issued
loan_data['loan_status'].count()


# In[52]:


#Loan proportion of each observation
loan_data['loan_status'].value_counts() / loan_data['loan_status'].count()


# In[53]:


# Good/ Bad loan.# We create a new variable that has the value of '0' if a condition is met (Default), and the value of '1' if it is not met (Non-default). 
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                       'Does not meet the credit policy. Status:Charged Off',
                                                       'Late (31-120 days)']), 0, 1)


# In[54]:


loan_data['good_bad']


# # Splitting Data into Train/Test

# In[55]:


# We split two dataframes with inputs and targets, each into a train and test dataframe, and store them in variables.
# We set the size of the test dataset to be 20% and the train dataset becomes 80%.
# We also set a specific random state.This would allow us to perform the exact same split multimple times.
# This means, to assign the exact same observations to the train and test datasets.
loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad', axis = 1), loan_data['good_bad'], test_size = 0.2, random_state = 42)


# In[56]:


# Displaying the size of the dataframes
print(loan_data_inputs_train.shape)
print(loan_data_targets_train.shape)
print(loan_data_inputs_test.shape)
print(loan_data_targets_test.shape)


# # DATA PREPROCESSING: TRAINING DATASET

# **Preprocessing More Discrete/ Categorical Variables**

# *Creating variables*
# 1. `n_obs` is total number of observations
# 2. `WOE` is Weight of Evidence
# 3. `n_good` is number of good loans
# 4. `n_bad` is number od bad loans
# 5. `prop_good` is proportion of good borrowers
# 6. `prop_bad` is proportion of bad borrowers
# 7. `prop_n_obs` is proportion of observations
# 8. `prop_n_good` is proportion of the number of good borrowers
# 9. `prop_n_bad` is proportion of the nuber of bad borrowers
# 10. `IV` is information value
# 11. `diff_prop_good` is difference of the proportion of good borrowers
# 

# In[57]:


df_inputs_prepr = loan_data_inputs_train
df_targets_prepr = loan_data_targets_train


# In[58]:


# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.
# WoE function for discrete unordered variables
def woe_discrete(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# In[59]:


#grade variable
#Executing the function and storing it in a dataframe
df_temp = woe_discrete(df_inputs_prepr, 'grade', df_targets_prepr)
df_temp


# The Informaton Value(IV) falls within 0.3 and 0.5, indicating strong predictive power. That is, `0.3<IV<0.5` 

# **Visualizing the Preprocessed variables**

# In[60]:


#We define a function that takes 2 arguments: a dataframe and a number.
def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    # Turns the values of the column with index 0 to strings, makes an array from these strings, and passes it to variable x.
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    # Selects a column with label 'WoE' and passes it to variable y.
    y = df_WoE['WoE']
    
    #Plotting the figure
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels)


# In[61]:


plot_by_woe(df_temp)


# The greater the grade, the greater the weight of evidence. That means loans with greater external ratings are greater on avaerage

# **Preprocessing Discrete Variables: Creating Dummy Variables**

# In[62]:


#home_ownership variable
#Executing our previous WOE function
df_temp1 = woe_discrete(df_inputs_prepr, 'home_ownership', df_targets_prepr)
df_temp1


# In[63]:


#Plotting the weight of evidence (woe) values by excuting the plot function we created previously.
plot_by_woe(df_temp1)


# In[64]:


# There are many categories with home_ownership variable.
# Therefore, we create a new discrete variable where we combine some of the categories.
# 'OTHER', 'ANY' and 'NONE' are riskiest but are very few. 'RENT' is the next riskiest.
# We combine them in one category, 'RENT_OTHER_NONE_ANY'.
# We end up with 3 categories for the 'home_onership': 'RENT_OTHER_NONE_ANY', 'OWN', 'MORTGAGE'.
df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_prepr['home_ownership:RENT'], df_inputs_prepr['home_ownership:OTHER'],
                                                      df_inputs_prepr['home_ownership:NONE'],df_inputs_prepr['home_ownership:ANY']])


# In[65]:


#Unique caterogies in the addr_state variable
df_inputs_prepr['addr_state'].unique()


# In[66]:


#addr_state variable
# We calculate weight of evidence.
df_temp2 = woe_discrete(df_inputs_prepr, 'addr_state', df_targets_prepr)
df_temp2


# In[67]:


# We plot the weight of evidence values.
plot_by_woe(df_temp2)


# In[68]:


#We want to get a normal curve
if ['addr_state:ND'] in df_inputs_prepr.columns.values:
    pass
else:
    df_inputs_prepr['addr_state:ND'] = 0


# In[69]:


# We plot the weight of evidence values again for the 'addr_state' by removing the state IA and AL.
plot_by_woe(df_temp2.iloc[2: -2, : ])


# In[70]:


# We plot the weight of evidence values again by removing the upper states OR, NH, DC, ID, VT, ME.
plot_by_woe(df_temp2.iloc[6: -6, : ])


# In[71]:


# Creating dummies for the 'addr_state' variable
# We create the following categories:
# 'ND' 'NE' 'IA' NV' 'FL' 'HI' 'AL'
# 'NM' 'VA'
# 'NY'
# 'OK' 'TN' 'MO' 'LA' 'MD' 'NC'
# 'CA'
# 'UT' 'KY' 'AZ' 'NJ'
# 'AR' 'MI' 'PA' 'OH' 'MN'
# 'RI' 'MA' 'DE' 'SD' 'IN'
# 'GA' 'WA' 'OR'
# 'WI' 'MT'
# 'TX'
# 'IL' 'CT'
# 'KS' 'SC' 'CO' 'VT' 'AK' 'MS'
# 'WV' 'NH' 'WY' 'DC' 'ME' 'ID'

# 'ND_NE_IA_NV_FL_HI_AL' will be the reference category.

df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prepr['addr_state:ND'], df_inputs_prepr['addr_state:NE'],
                                              df_inputs_prepr['addr_state:IA'], df_inputs_prepr['addr_state:NV'],
                                              df_inputs_prepr['addr_state:FL'], df_inputs_prepr['addr_state:HI'],
                                                          df_inputs_prepr['addr_state:AL']])

df_inputs_prepr['addr_state:NM_VA'] = sum([df_inputs_prepr['addr_state:NM'], df_inputs_prepr['addr_state:VA']])

df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prepr['addr_state:OK'], df_inputs_prepr['addr_state:TN'],
                                              df_inputs_prepr['addr_state:MO'], df_inputs_prepr['addr_state:LA'],
                                              df_inputs_prepr['addr_state:MD'], df_inputs_prepr['addr_state:NC']])

df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = sum([df_inputs_prepr['addr_state:UT'], df_inputs_prepr['addr_state:KY'],
                                              df_inputs_prepr['addr_state:AZ'], df_inputs_prepr['addr_state:NJ']])

df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prepr['addr_state:AR'], df_inputs_prepr['addr_state:MI'],
                                              df_inputs_prepr['addr_state:PA'], df_inputs_prepr['addr_state:OH'],
                                              df_inputs_prepr['addr_state:MN']])

df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prepr['addr_state:RI'], df_inputs_prepr['addr_state:MA'],
                                              df_inputs_prepr['addr_state:DE'], df_inputs_prepr['addr_state:SD'],
                                              df_inputs_prepr['addr_state:IN']])

df_inputs_prepr['addr_state:GA_WA_OR'] = sum([df_inputs_prepr['addr_state:GA'], df_inputs_prepr['addr_state:WA'],
                                              df_inputs_prepr['addr_state:OR']])

df_inputs_prepr['addr_state:WI_MT'] = sum([df_inputs_prepr['addr_state:WI'], df_inputs_prepr['addr_state:MT']])

df_inputs_prepr['addr_state:IL_CT'] = sum([df_inputs_prepr['addr_state:IL'], df_inputs_prepr['addr_state:CT']])

df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prepr['addr_state:KS'], df_inputs_prepr['addr_state:SC'],
                                              df_inputs_prepr['addr_state:CO'], df_inputs_prepr['addr_state:VT'],
                                              df_inputs_prepr['addr_state:AK'], df_inputs_prepr['addr_state:MS']])

df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prepr['addr_state:WV'], df_inputs_prepr['addr_state:NH'],
                                              df_inputs_prepr['addr_state:WY'], df_inputs_prepr['addr_state:DC'],
                                              df_inputs_prepr['addr_state:ME'], df_inputs_prepr['addr_state:ID']])


# In[72]:


# 'verification_status' variable'
# We calculate weight of evidence.

df_temp3 = woe_discrete(df_inputs_prepr, 'verification_status', df_targets_prepr)
df_temp3


# In[73]:


# We plot the weight of evidence values.
plot_by_woe(df_temp3)


# In[74]:


# 'purpose' variable
# We calculate weight of evidence.

df_temp4 = woe_discrete(df_inputs_prepr, 'purpose', df_targets_prepr)
df_temp4


# In[75]:


plot_by_woe(df_temp4, 90)
# We plot the weight of evidence values.


# In[76]:


# We create dummy variables for the 'purpose' variable
# We combine 'educational', 'small_business', 'wedding', 'renewable_energy', 'moving', 'house' in one category: 'educ__sm_b__wedd__ren_en__mov__house'.
# We combine 'other', 'medical', 'vacation' in one category: 'oth__med__vacation'.
# We combine 'major_purchase', 'car', 'home_improvement' in one category: 'major_purch__car__home_impr'.
# We leave 'debt_consolidtion' in a separate category.
# We leave 'credit_card' in a separate category.
# 'educ__sm_b__wedd__ren_en__mov__house' will be the reference category.
df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prepr['purpose:educational'], df_inputs_prepr['purpose:small_business'],
                                                                 df_inputs_prepr['purpose:wedding'], df_inputs_prepr['purpose:renewable_energy'],
                                                                 df_inputs_prepr['purpose:moving'], df_inputs_prepr['purpose:house']])
df_inputs_prepr['purpose:oth__med__vacation'] = sum([df_inputs_prepr['purpose:other'], df_inputs_prepr['purpose:medical'],
                                             df_inputs_prepr['purpose:vacation']])
df_inputs_prepr['purpose:major_purch__car__home_impr'] = sum([df_inputs_prepr['purpose:major_purchase'], df_inputs_prepr['purpose:car'],
                                                        df_inputs_prepr['purpose:home_improvement']])


# In[77]:


# 'initial_list_status' variable
# We calculate weight of evidence.

df_temp5 = woe_discrete(df_inputs_prepr, 'initial_list_status', df_targets_prepr)
df_temp5


# In[78]:


# We plot the weight of evidence values.
plot_by_woe(df_temp5)


# **Preprocessing Continuous Variables: Creating dummy variables**

# In[79]:


# WoE function for ordered discrete and continuous variables
# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.

def woe_ordered_continuous(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# In[80]:


# term variable
# There are only two unique values, 36 and 60.
df_inputs_prepr['term_int'].unique()


# In[81]:


# We calculate weight of evidence.
df_temp6 = woe_ordered_continuous(df_inputs_prepr, 'term_int', df_targets_prepr)
df_temp6


# In[82]:


# We plot the weight of evidence values.
plot_by_woe(df_temp6)


# It seems 60 months loans are much risky than 36 months loans

# In[83]:


# We will keep both the 36 and 60 months category.
# However the '60' months will be the reference category.
df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)


# In[84]:


# emp_length_int variable
# Has only 11 levels: from 0 to 10. Hence, we turn it into a factor with 11 levels.
df_inputs_prepr['emp_length_int'].unique()


# In[85]:


# We calculate weight of evidence.
df_temp7 = woe_ordered_continuous(df_inputs_prepr, 'emp_length_int', df_targets_prepr)
df_temp7


# In[86]:


# We plot the weight of evidence values.
plot_by_woe(df_temp7)


# In[87]:


# Employment length has several categories
# So we have to create the following new categories for emp_length_int: '0', '1', '2 - 4', '5 - 6', '7 - 9', '10'
# '0' will be the reference category
df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
df_inputs_prepr['emp_length:1'] = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)


# In[88]:


# Months since loan issue date (mths_since_issue_d) variable
df_inputs_prepr['mths_since_issue_d'].unique()


# In[89]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['mths_since_issue_d_factor'] = pd.cut(df_inputs_prepr['mths_since_issue_d'], 50)


# In[90]:


df_inputs_prepr['mths_since_issue_d_factor']


# In[91]:


# mths_since_issue_d
# We calculate weight of evidence.
df_temp8 = woe_ordered_continuous(df_inputs_prepr, 'mths_since_issue_d_factor', df_targets_prepr)
df_temp8


# In[92]:


# We plot the weight of evidence values.
plot_by_woe(df_temp8)


# We have to rotate the labels because we cannot read them otherwise.
# 

# In[93]:


# We plot the weight of evidence values, rotating the labels 90 degrees.
plot_by_woe(df_temp8, 90)


# In[94]:


# We plot the weight of evidence values.
plot_by_woe(df_temp8.iloc[3: , : ], 90)


# In[95]:


# We create the following categories:
# < 38, 38 - 39, 40 - 41, 42 - 48, 49 - 52, 53 - 64, 65 - 84, > 84.
df_inputs_prepr['mths_since_issue_d:<38'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38)), 1, 0)
df_inputs_prepr['mths_since_issue_d:38-39'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
df_inputs_prepr['mths_since_issue_d:40-41'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
df_inputs_prepr['mths_since_issue_d:42-48'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
df_inputs_prepr['mths_since_issue_d:49-52'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
df_inputs_prepr['mths_since_issue_d:53-64'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
df_inputs_prepr['mths_since_issue_d:65-84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(85, int(df_inputs_prepr['mths_since_issue_d'].max()))), 1, 0)


# In[96]:


# int_rate variable
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)


# In[97]:


# We calculate weight of evidence.
df_temp9 = woe_ordered_continuous(df_inputs_prepr, 'int_rate_factor', df_targets_prepr)
df_temp9


# In[98]:


# We plot the weight of evidence values.
plot_by_woe(df_temp9, 90)


# In[99]:


# We create the following categories:
# '< 9.548', '9.548 - 12.025', '12.025 - 15.74', '15.74 - 20.281', '> 20.281'
df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] <= 9.548), 1, 0)
df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] <= 12.025), 1, 0)
df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74), 1, 0)
df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] <= 20.281), 1, 0)
df_inputs_prepr['int_rate:>20.281'] = np.where((df_inputs_prepr['int_rate'] > 20.281), 1, 0)


# In[100]:


# funded_amnt variable
df_inputs_prepr['funded_amnt'].unique()


# In[101]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['funded_amnt_factor'] = pd.cut(df_inputs_prepr['funded_amnt'], 50)


# In[102]:


# We calculate weight of evidence.
df_temp10 = woe_ordered_continuous(df_inputs_prepr, 'funded_amnt_factor', df_targets_prepr)
df_temp10


# In[103]:


# We plot the weight of evidence values.
plot_by_woe(df_temp10, 90)


# In[104]:


# mths_since_earliest_cr_line variable
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_prepr['mths_since_earliest_cr_line'], 50)


# In[105]:


# We calculate weight of evidence.
df_temp11 = woe_ordered_continuous(df_inputs_prepr, 'mths_since_earliest_cr_line_factor', df_targets_prepr)
df_temp11


# In[106]:


# We plot the weight of evidence values.
plot_by_woe(df_temp11, 90)


# In[107]:


# We plot the weight of evidence values
plot_by_woe(df_temp11.iloc[6: , : ], 90)


# In[108]:


# We create the following categories:
# < 140, # 141 - 164, # 165 - 247, # 248 - 270, # 271 - 352, # > 352
df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)


# In[109]:


# delinq_2yrs variable
# We calculate weight of evidence
df_temp12 = woe_ordered_continuous(df_inputs_prepr, 'delinq_2yrs', df_targets_prepr)
df_temp12


# In[110]:


# We plot the weight of evidence values
plot_by_woe(df_temp12)


# In[111]:


# We create the following Categories: 0, 1-3, >=4
df_inputs_prepr['delinq_2yrs:0'] = np.where((df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
df_inputs_prepr['delinq_2yrs:>=4'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 9), 1, 0)


# In[112]:


# inq_last_6mths variable
# We calculate weight of evidence.
df_temp13 = woe_ordered_continuous(df_inputs_prepr, 'inq_last_6mths', df_targets_prepr)
df_temp13


# In[113]:


# We plot the weight of evidence values
plot_by_woe(df_temp13)


# In[114]:


# We create the following Categories: 0, 1 - 2, 3 - 6, > 6
df_inputs_prepr['inq_last_6mths:0'] = np.where((df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
df_inputs_prepr['inq_last_6mths:>6'] = np.where((df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)


# In[115]:


# open_acc variable
# We calculate weight of evidence.
df_temp14 = woe_ordered_continuous(df_inputs_prepr, 'open_acc', df_targets_prepr)
df_temp14


# In[116]:


# We plot the weight of evidence values
plot_by_woe(df_temp14, 90)


# In[117]:


# We plot the weight of evidence values
plot_by_woe(df_temp14.iloc[ : 40, :], 90)


# In[118]:


# We create the following Categories: '0', '1-3', '4-12', '13-17', '18-22', '23-25', '26-30', '>30'
df_inputs_prepr['open_acc:0'] = np.where((df_inputs_prepr['open_acc'] == 0), 1, 0)
df_inputs_prepr['open_acc:1-3'] = np.where((df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
df_inputs_prepr['open_acc:4-12'] = np.where((df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
df_inputs_prepr['open_acc:>=31'] = np.where((df_inputs_prepr['open_acc'] >= 31), 1, 0)


# In[119]:


# pub_rec variable
# We calculate weight of evidence
df_temp15 = woe_ordered_continuous(df_inputs_prepr, 'pub_rec', df_targets_prepr)
df_temp15


# In[120]:


# We plot the weight of evidence values.
plot_by_woe(df_temp15, 90)


# In[121]:


# We create the following Categories: '0-2', '3-4', '>=5'
df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
df_inputs_prepr['pub_rec:>=5'] = np.where((df_inputs_prepr['pub_rec'] >= 5), 1, 0)


# In[122]:


# total_acc variable
df_inputs_prepr['total_acc'].unique()


# In[123]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)


# In[124]:


# We calculate weight of evidence
df_temp16 = woe_ordered_continuous(df_inputs_prepr, 'total_acc_factor', df_targets_prepr)
df_temp16


# In[125]:


# We plot the weight of evidence values
plot_by_woe(df_temp16, 90)


# In[126]:


# We create the following Categories: '<=27', '28-51', '>51'
df_inputs_prepr['total_acc:<=27'] = np.where((df_inputs_prepr['total_acc'] <= 27), 1, 0)
df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
df_inputs_prepr['total_acc:>=52'] = np.where((df_inputs_prepr['total_acc'] >= 52), 1, 0)


# In[127]:


# acc_now_delinq variable
# We calculate weight of evidence.
df_temp17 = woe_ordered_continuous(df_inputs_prepr, 'acc_now_delinq', df_targets_prepr)
df_temp17


# In[128]:


# We plot the weight of evidence values
plot_by_woe(df_temp17)


# In[129]:


# We create the following Categories: '0', '>=1'
df_inputs_prepr['acc_now_delinq:0'] = np.where((df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
df_inputs_prepr['acc_now_delinq:>=1'] = np.where((df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)


# In[130]:


# total_rev_hi_lim variable
df_inputs_prepr['total_rev_hi_lim'].unique()


# In[131]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 2000 categories by its values.
df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)


# In[132]:


# We calculate weight of evidence.
df_temp18 = woe_ordered_continuous(df_inputs_prepr, 'total_rev_hi_lim_factor', df_targets_prepr)
df_temp18


# In[133]:


# We plot the weight of evidence values.
plot_by_woe(df_temp18.iloc[: 50, : ], 90)


# In[134]:


# We create the following Categories:'<=5K', '5K-10K', '10K-20K', '20K-30K', '30K-40K', '40K-55K', '55K-95K', '>95K'
df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)


# In[135]:


# installment variable
df_inputs_prepr['installment'].unique()


# In[136]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)


# In[137]:


# We calculate weight of evidence.
df_temp19 = woe_ordered_continuous(df_inputs_prepr, 'installment_factor', df_targets_prepr)
df_temp19


# In[138]:


# We plot the weight of evidence values.
plot_by_woe(df_temp19, 90)


# In[139]:


# annual_inc variable
df_inputs_prepr['annual_inc'].unique()


# In[140]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 50)

# We calculate weight of evidence.
df_temp20 = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_targets_prepr)
df_temp20


# In[141]:


# Splitting the initial ‘annual income’ variable into 50 categories doesn't work well for fine classing because there are a lot of people with low income and very few people with high income.
# Thus, we do fine-classing using the 'cut' method, we split the variable into 100 categories by its values.
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)

# We calculate weight of evidence.
df_temp20 = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_targets_prepr)
df_temp20


# In[142]:


# Initial examination shows that there are too few individuals with large income and too many with small income.
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, : ]


# In[143]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)

# We calculate weight of evidence.
df_temp20 = woe_ordered_continuous(df_inputs_prepr_temp, 'annual_inc_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp20


# In[144]:


# We plot the weight of evidence values.
plot_by_woe(df_temp20, 90)


# In[145]:


# WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)


# In[146]:


# mths_since_last_delinq variable
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])]
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50)


# We calculate weight of evidence.
df_temp21 = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp21


# In[147]:


# We plot the weight of evidence values.
plot_by_woe(df_temp21, 90)


# In[148]:


# We create the following Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)


# In[149]:


# dti variable
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 100)

# We calculate weight of evidence.
df_temp22 = woe_ordered_continuous(df_inputs_prepr, 'dti_factor', df_targets_prepr)
df_temp22


# In[150]:


# We plot the weight of evidence values.
plot_by_woe(df_temp22, 90)


# In[151]:


# Similarly to income, initial examination shows that most values are lower than 200.
# Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
# the categories of everyone with 150k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['dti'] <= 35, : ]


# In[152]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr_temp['dti'], 50)


# We calculate weight of evidence.
df_temp22 = woe_ordered_continuous(df_inputs_prepr_temp, 'dti_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp22


# In[153]:


# We plot the weight of evidence values.
plot_by_woe(df_temp22, 90)


# In[154]:


# We create the following Categories:
df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
df_inputs_prepr['dti:1.4-3.5'] = np.where((df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
df_inputs_prepr['dti:3.5-7.7'] = np.where((df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
df_inputs_prepr['dti:7.7-10.5'] = np.where((df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
df_inputs_prepr['dti:22.4-35'] = np.where((df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)


# In[155]:


# mths_since_last_record variable
# We have to create one category for missing values and do fine and coarse classing for the rest.
#sum(loan_data_temp['mths_since_last_record'].isnull())

df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_record'])]

# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr_temp['mths_since_last_record_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_record'], 50)

# We calculate weight of evidence.
df_temp23 = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_record_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp23


# In[156]:


# We plot the weight of evidence values.
plot_by_woe(df_temp23, 90)


# In[157]:


# We create the following Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_prepr['mths_since_last_record:Missing'] = np.where((df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_record:0-2'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
df_inputs_prepr['mths_since_last_record:3-20'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
df_inputs_prepr['mths_since_last_record:21-31'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
df_inputs_prepr['mths_since_last_record:32-80'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
df_inputs_prepr['mths_since_last_record:81-86'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
df_inputs_prepr['mths_since_last_record:>86'] = np.where((df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)


# # DATA PREPROCESSING: TEST DATASET

# **We will do the same like we did for the train dataset above**

# In[158]:


df_inputs_prepr = loan_data_inputs_test
df_targets_prepr = loan_data_targets_test


# In[159]:


# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.
# WoE function for discrete unordered variables
def woe_discrete(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# In[160]:


#grade variable
#Executing the function and storing it in a dataframe
df_temp = woe_discrete(df_inputs_prepr, 'grade', df_targets_prepr)
df_temp


# In[161]:


#We define a function that takes 2 arguments: a dataframe and a number.
def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    # Turns the values of the column with index 0 to strings, makes an array from these strings, and passes it to variable x.
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    # Selects a column with label 'WoE' and passes it to variable y.
    y = df_WoE['WoE']
    
    #Plotting the figure
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels)


# In[162]:


plot_by_woe(df_temp)


# The greater the grade, the greater the weight of evidence. That means loans with greater external ratings are greater on avaerage

# In[163]:


#home_ownership variable
#Executing our previous WOE function
df_temp1 = woe_discrete(df_inputs_prepr, 'home_ownership', df_targets_prepr)
df_temp1


# In[164]:


#Plotting the weight of evidence (woe) values by excuting the plot function we created previously.
plot_by_woe(df_temp1)


# In[165]:


# There are many categories with home_ownership variable.
# Therefore, we create a new discrete variable where we combine some of the categories.
# 'OTHER', 'ANY' and 'NONE' are riskiest but are very few. 'RENT' is the next riskiest.
# We combine them in one category, 'RENT_OTHER_NONE_ANY'.
# We end up with 3 categories for the 'home_onership': 'RENT_OTHER_NONE_ANY', 'OWN', 'MORTGAGE'.
df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_prepr['home_ownership:RENT'], df_inputs_prepr['home_ownership:OTHER'],
                                                      df_inputs_prepr['home_ownership:NONE'],df_inputs_prepr['home_ownership:ANY']])


# In[166]:


#Unique caterogies in the addr_state variable
df_inputs_prepr['addr_state'].unique()


# In[167]:


#addr_state variable
# We calculate weight of evidence.
df_temp2 = woe_discrete(df_inputs_prepr, 'addr_state', df_targets_prepr)
df_temp2


# In[168]:


# We plot the weight of evidence values.
plot_by_woe(df_temp2)


# In[169]:


#We want to get a normal curve
if ['addr_state:ND'] in df_inputs_prepr.columns.values:
    pass
else:
    df_inputs_prepr['addr_state:ND'] = 0


# In[170]:


# We plot the weight of evidence values again for the 'addr_state' by removing the state IA and AL.
plot_by_woe(df_temp2.iloc[2: -2, : ])


# In[171]:


# We plot the weight of evidence values again by removing the upper states OR, NH, DC, ID, VT, ME.
plot_by_woe(df_temp2.iloc[6: -6, : ])


# In[172]:


# Creating dummies for the 'addr_state' variable
# We create the following categories:
# 'ND' 'NE' 'IA' NV' 'FL' 'HI' 'AL'
# 'NM' 'VA'
# 'NY'
# 'OK' 'TN' 'MO' 'LA' 'MD' 'NC'
# 'CA'
# 'UT' 'KY' 'AZ' 'NJ'
# 'AR' 'MI' 'PA' 'OH' 'MN'
# 'RI' 'MA' 'DE' 'SD' 'IN'
# 'GA' 'WA' 'OR'
# 'WI' 'MT'
# 'TX'
# 'IL' 'CT'
# 'KS' 'SC' 'CO' 'VT' 'AK' 'MS'
# 'WV' 'NH' 'WY' 'DC' 'ME' 'ID'

# 'ND_NE_IA_NV_FL_HI_AL' will be the reference category.

df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prepr['addr_state:ND'], df_inputs_prepr['addr_state:NE'],
                                              df_inputs_prepr['addr_state:IA'], df_inputs_prepr['addr_state:NV'],
                                              df_inputs_prepr['addr_state:FL'], df_inputs_prepr['addr_state:HI'],
                                                          df_inputs_prepr['addr_state:AL']])

df_inputs_prepr['addr_state:NM_VA'] = sum([df_inputs_prepr['addr_state:NM'], df_inputs_prepr['addr_state:VA']])

df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prepr['addr_state:OK'], df_inputs_prepr['addr_state:TN'],
                                              df_inputs_prepr['addr_state:MO'], df_inputs_prepr['addr_state:LA'],
                                              df_inputs_prepr['addr_state:MD'], df_inputs_prepr['addr_state:NC']])

df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = sum([df_inputs_prepr['addr_state:UT'], df_inputs_prepr['addr_state:KY'],
                                              df_inputs_prepr['addr_state:AZ'], df_inputs_prepr['addr_state:NJ']])

df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prepr['addr_state:AR'], df_inputs_prepr['addr_state:MI'],
                                              df_inputs_prepr['addr_state:PA'], df_inputs_prepr['addr_state:OH'],
                                              df_inputs_prepr['addr_state:MN']])

df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prepr['addr_state:RI'], df_inputs_prepr['addr_state:MA'],
                                              df_inputs_prepr['addr_state:DE'], df_inputs_prepr['addr_state:SD'],
                                              df_inputs_prepr['addr_state:IN']])

df_inputs_prepr['addr_state:GA_WA_OR'] = sum([df_inputs_prepr['addr_state:GA'], df_inputs_prepr['addr_state:WA'],
                                              df_inputs_prepr['addr_state:OR']])

df_inputs_prepr['addr_state:WI_MT'] = sum([df_inputs_prepr['addr_state:WI'], df_inputs_prepr['addr_state:MT']])

df_inputs_prepr['addr_state:IL_CT'] = sum([df_inputs_prepr['addr_state:IL'], df_inputs_prepr['addr_state:CT']])

df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prepr['addr_state:KS'], df_inputs_prepr['addr_state:SC'],
                                              df_inputs_prepr['addr_state:CO'], df_inputs_prepr['addr_state:VT'],
                                              df_inputs_prepr['addr_state:AK'], df_inputs_prepr['addr_state:MS']])

df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prepr['addr_state:WV'], df_inputs_prepr['addr_state:NH'],
                                              df_inputs_prepr['addr_state:WY'], df_inputs_prepr['addr_state:DC'],
                                              df_inputs_prepr['addr_state:ME'], df_inputs_prepr['addr_state:ID']])


# In[173]:


# 'verification_status' variable'
# We calculate weight of evidence.

df_temp3 = woe_discrete(df_inputs_prepr, 'verification_status', df_targets_prepr)
df_temp3


# In[174]:


# We plot the weight of evidence values.
plot_by_woe(df_temp3)


# In[175]:


# 'purpose' variable
# We calculate weight of evidence.

df_temp4 = woe_discrete(df_inputs_prepr, 'purpose', df_targets_prepr)
df_temp4


# In[176]:


plot_by_woe(df_temp4, 90)
# We plot the weight of evidence values.


# In[177]:


# We create dummy variables for the 'purpose' variable
# We combine 'educational', 'small_business', 'wedding', 'renewable_energy', 'moving', 'house' in one category: 'educ__sm_b__wedd__ren_en__mov__house'.
# We combine 'other', 'medical', 'vacation' in one category: 'oth__med__vacation'.
# We combine 'major_purchase', 'car', 'home_improvement' in one category: 'major_purch__car__home_impr'.
# We leave 'debt_consolidtion' in a separate category.
# We leave 'credit_card' in a separate category.
# 'educ__sm_b__wedd__ren_en__mov__house' will be the reference category.
df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prepr['purpose:educational'], df_inputs_prepr['purpose:small_business'],
                                                                 df_inputs_prepr['purpose:wedding'], df_inputs_prepr['purpose:renewable_energy'],
                                                                 df_inputs_prepr['purpose:moving'], df_inputs_prepr['purpose:house']])
df_inputs_prepr['purpose:oth__med__vacation'] = sum([df_inputs_prepr['purpose:other'], df_inputs_prepr['purpose:medical'],
                                             df_inputs_prepr['purpose:vacation']])
df_inputs_prepr['purpose:major_purch__car__home_impr'] = sum([df_inputs_prepr['purpose:major_purchase'], df_inputs_prepr['purpose:car'],
                                                        df_inputs_prepr['purpose:home_improvement']])


# In[178]:


# 'initial_list_status' variable
# We calculate weight of evidence.

df_temp5 = woe_discrete(df_inputs_prepr, 'initial_list_status', df_targets_prepr)
df_temp5


# In[179]:


# We plot the weight of evidence values.
plot_by_woe(df_temp5)


# In[180]:


# WoE function for ordered discrete and continuous variables
# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.

def woe_ordered_continuous(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# In[181]:


# term variable
# There are only two unique values, 36 and 60.
df_inputs_prepr['term_int'].unique()


# In[182]:


# We calculate weight of evidence.
df_temp6 = woe_ordered_continuous(df_inputs_prepr, 'term_int', df_targets_prepr)
df_temp6


# In[183]:


# We plot the weight of evidence values.
plot_by_woe(df_temp6)


# It seems 60 months loans are much risky than 36 months loans

# In[184]:


# We will keep both the 36 and 60 months category.
# However the '60' months will be the reference category.
df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)


# In[185]:


# emp_length_int variable
# Has only 11 levels: from 0 to 10. Hence, we turn it into a factor with 11 levels.
df_inputs_prepr['emp_length_int'].unique()


# In[186]:


# We calculate weight of evidence.
df_temp7 = woe_ordered_continuous(df_inputs_prepr, 'emp_length_int', df_targets_prepr)
df_temp7


# In[187]:


# We plot the weight of evidence values.
plot_by_woe(df_temp7)


# In[188]:


# Employment length has several categories
# So we have to create the following new categories for emp_length_int: '0', '1', '2 - 4', '5 - 6', '7 - 9', '10'
# '0' will be the reference category
df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
df_inputs_prepr['emp_length:1'] = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)


# In[189]:


# Months since loan issue date (mths_since_issue_d) variable
df_inputs_prepr['mths_since_issue_d'].unique()


# In[190]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['mths_since_issue_d_factor'] = pd.cut(df_inputs_prepr['mths_since_issue_d'], 50)


# In[191]:


df_inputs_prepr['mths_since_issue_d_factor']


# In[192]:


# mths_since_issue_d
# We calculate weight of evidence.
df_temp8 = woe_ordered_continuous(df_inputs_prepr, 'mths_since_issue_d_factor', df_targets_prepr)
df_temp8


# In[193]:


# We plot the weight of evidence values.
plot_by_woe(df_temp8)


# We have to rotate the labels because we cannot read them otherwise.
# 

# In[194]:


# We plot the weight of evidence values, rotating the labels 90 degrees.
plot_by_woe(df_temp8, 90)


# In[195]:


# We plot the weight of evidence values.
plot_by_woe(df_temp8.iloc[3: , : ], 90)


# In[196]:


# We create the following categories:
# < 38, 38 - 39, 40 - 41, 42 - 48, 49 - 52, 53 - 64, 65 - 84, > 84.
df_inputs_prepr['mths_since_issue_d:<38'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38)), 1, 0)
df_inputs_prepr['mths_since_issue_d:38-39'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
df_inputs_prepr['mths_since_issue_d:40-41'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
df_inputs_prepr['mths_since_issue_d:42-48'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
df_inputs_prepr['mths_since_issue_d:49-52'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
df_inputs_prepr['mths_since_issue_d:53-64'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
df_inputs_prepr['mths_since_issue_d:65-84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(85, int(df_inputs_prepr['mths_since_issue_d'].max()))), 1, 0)


# In[197]:


# int_rate variable
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)


# In[198]:


# We calculate weight of evidence.
df_temp9 = woe_ordered_continuous(df_inputs_prepr, 'int_rate_factor', df_targets_prepr)
df_temp9


# In[199]:


# We plot the weight of evidence values.
plot_by_woe(df_temp9, 90)


# In[200]:


# We create the following categories:
# '< 9.548', '9.548 - 12.025', '12.025 - 15.74', '15.74 - 20.281', '> 20.281'
df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] <= 9.548), 1, 0)
df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] <= 12.025), 1, 0)
df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74), 1, 0)
df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] <= 20.281), 1, 0)
df_inputs_prepr['int_rate:>20.281'] = np.where((df_inputs_prepr['int_rate'] > 20.281), 1, 0)


# In[201]:


# funded_amnt variable
df_inputs_prepr['funded_amnt'].unique()


# In[202]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['funded_amnt_factor'] = pd.cut(df_inputs_prepr['funded_amnt'], 50)


# In[203]:


# We calculate weight of evidence.
df_temp10 = woe_ordered_continuous(df_inputs_prepr, 'funded_amnt_factor', df_targets_prepr)
df_temp10


# In[204]:


# We plot the weight of evidence values.
plot_by_woe(df_temp10, 90)


# In[205]:


# mths_since_earliest_cr_line variable
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_prepr['mths_since_earliest_cr_line'], 50)


# In[206]:


# We calculate weight of evidence.
df_temp11 = woe_ordered_continuous(df_inputs_prepr, 'mths_since_earliest_cr_line_factor', df_targets_prepr)
df_temp11


# In[207]:


# We plot the weight of evidence values.
plot_by_woe(df_temp11, 90)


# In[208]:


# We plot the weight of evidence values
plot_by_woe(df_temp11.iloc[6: , : ], 90)


# In[209]:


# We create the following categories:
# < 140, # 141 - 164, # 165 - 247, # 248 - 270, # 271 - 352, # > 352
df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)


# In[210]:


# delinq_2yrs variable
# We calculate weight of evidence
df_temp12 = woe_ordered_continuous(df_inputs_prepr, 'delinq_2yrs', df_targets_prepr)
df_temp12


# In[211]:


# We plot the weight of evidence values
plot_by_woe(df_temp12)


# In[212]:


# We create the following Categories: 0, 1-3, >=4
df_inputs_prepr['delinq_2yrs:0'] = np.where((df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
df_inputs_prepr['delinq_2yrs:>=4'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 9), 1, 0)


# In[213]:


# inq_last_6mths variable
# We calculate weight of evidence.
df_temp13 = woe_ordered_continuous(df_inputs_prepr, 'inq_last_6mths', df_targets_prepr)
df_temp13


# In[214]:


# We plot the weight of evidence values
plot_by_woe(df_temp13)


# In[215]:


# We create the following Categories: 0, 1 - 2, 3 - 6, > 6
df_inputs_prepr['inq_last_6mths:0'] = np.where((df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
df_inputs_prepr['inq_last_6mths:>6'] = np.where((df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)


# In[216]:


# open_acc variable
# We calculate weight of evidence.
df_temp14 = woe_ordered_continuous(df_inputs_prepr, 'open_acc', df_targets_prepr)
df_temp14


# In[217]:


# We plot the weight of evidence values
plot_by_woe(df_temp14, 90)


# In[218]:


# We plot the weight of evidence values
plot_by_woe(df_temp14.iloc[ : 40, :], 90)


# In[219]:


# We create the following Categories: '0', '1-3', '4-12', '13-17', '18-22', '23-25', '26-30', '>30'
df_inputs_prepr['open_acc:0'] = np.where((df_inputs_prepr['open_acc'] == 0), 1, 0)
df_inputs_prepr['open_acc:1-3'] = np.where((df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
df_inputs_prepr['open_acc:4-12'] = np.where((df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
df_inputs_prepr['open_acc:>=31'] = np.where((df_inputs_prepr['open_acc'] >= 31), 1, 0)


# In[220]:


# pub_rec variable
# We calculate weight of evidence
df_temp15 = woe_ordered_continuous(df_inputs_prepr, 'pub_rec', df_targets_prepr)
df_temp15


# In[221]:


# We plot the weight of evidence values.
plot_by_woe(df_temp15, 90)


# In[222]:


# We create the following Categories: '0-2', '3-4', '>=5'
df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
df_inputs_prepr['pub_rec:>=5'] = np.where((df_inputs_prepr['pub_rec'] >= 5), 1, 0)


# In[223]:


# total_acc variable
df_inputs_prepr['total_acc'].unique()


# In[224]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)


# In[225]:


# We calculate weight of evidence
df_temp16 = woe_ordered_continuous(df_inputs_prepr, 'total_acc_factor', df_targets_prepr)
df_temp16


# In[226]:


# We plot the weight of evidence values
plot_by_woe(df_temp16, 90)


# In[227]:


# We create the following Categories: '<=27', '28-51', '>51'
df_inputs_prepr['total_acc:<=27'] = np.where((df_inputs_prepr['total_acc'] <= 27), 1, 0)
df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
df_inputs_prepr['total_acc:>=52'] = np.where((df_inputs_prepr['total_acc'] >= 52), 1, 0)


# In[228]:


# acc_now_delinq variable
# We calculate weight of evidence.
df_temp17 = woe_ordered_continuous(df_inputs_prepr, 'acc_now_delinq', df_targets_prepr)
df_temp17


# In[229]:


# We plot the weight of evidence values
plot_by_woe(df_temp17)


# In[230]:


# We create the following Categories: '0', '>=1'
df_inputs_prepr['acc_now_delinq:0'] = np.where((df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
df_inputs_prepr['acc_now_delinq:>=1'] = np.where((df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)


# In[231]:


# total_rev_hi_lim variable
df_inputs_prepr['total_rev_hi_lim'].unique()


# In[232]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 2000 categories by its values.
df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)


# In[233]:


# We calculate weight of evidence.
df_temp18 = woe_ordered_continuous(df_inputs_prepr, 'total_rev_hi_lim_factor', df_targets_prepr)
df_temp18


# In[234]:


# We plot the weight of evidence values.
plot_by_woe(df_temp18.iloc[: 50, : ], 90)


# In[235]:


# We create the following Categories:'<=5K', '5K-10K', '10K-20K', '20K-30K', '30K-40K', '40K-55K', '55K-95K', '>95K'
df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)


# In[236]:


# installment variable
df_inputs_prepr['installment'].unique()


# In[237]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)


# In[238]:


# We calculate weight of evidence.
df_temp19 = woe_ordered_continuous(df_inputs_prepr, 'installment_factor', df_targets_prepr)
df_temp19


# In[239]:


# We plot the weight of evidence values.
plot_by_woe(df_temp19, 90)


# In[240]:


# annual_inc variable
df_inputs_prepr['annual_inc'].unique()


# In[241]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 50)

# We calculate weight of evidence.
df_temp20 = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_targets_prepr)
df_temp20


# In[242]:


# Splitting the initial ‘annual income’ variable into 50 categories doesn't work well for fine classing because there are a lot of people with low income and very few people with high income.
# Thus, we do fine-classing using the 'cut' method, we split the variable into 100 categories by its values.
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)

# We calculate weight of evidence.
df_temp20 = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_targets_prepr)
df_temp20


# In[243]:


# Initial examination shows that there are too few individuals with large income and too many with small income.
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, : ]


# In[244]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)

# We calculate weight of evidence.
df_temp20 = woe_ordered_continuous(df_inputs_prepr_temp, 'annual_inc_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp20


# In[245]:


# We plot the weight of evidence values.
plot_by_woe(df_temp20, 90)


# In[246]:


# WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)


# In[247]:


# mths_since_last_delinq variable
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])]
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50)


# We calculate weight of evidence.
df_temp21 = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp21


# In[248]:


# We plot the weight of evidence values.
plot_by_woe(df_temp21, 90)


# In[249]:


# We create the following Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)


# In[250]:


# dti variable
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 100)

# We calculate weight of evidence.
df_temp22 = woe_ordered_continuous(df_inputs_prepr, 'dti_factor', df_targets_prepr)
df_temp22


# In[251]:


# We plot the weight of evidence values.
plot_by_woe(df_temp22, 90)


# In[252]:


# Similarly to income, initial examination shows that most values are lower than 200.
# Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
# the categories of everyone with 150k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['dti'] <= 35, : ]


# In[253]:


# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr_temp['dti'], 50)


# We calculate weight of evidence.
df_temp22 = woe_ordered_continuous(df_inputs_prepr_temp, 'dti_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp22


# In[254]:


# We plot the weight of evidence values.
plot_by_woe(df_temp22, 90)


# In[255]:


# We create the following Categories:
df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
df_inputs_prepr['dti:1.4-3.5'] = np.where((df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
df_inputs_prepr['dti:3.5-7.7'] = np.where((df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
df_inputs_prepr['dti:7.7-10.5'] = np.where((df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
df_inputs_prepr['dti:22.4-35'] = np.where((df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)


# In[256]:


# mths_since_last_record variable
# We have to create one category for missing values and do fine and coarse classing for the rest.
#sum(loan_data_temp['mths_since_last_record'].isnull())

df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_record'])]

# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_prepr_temp['mths_since_last_record_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_record'], 50)

# We calculate weight of evidence.
df_temp23 = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_record_factor', df_targets_prepr[df_inputs_prepr_temp.index])
df_temp23


# In[257]:


# We plot the weight of evidence values.
plot_by_woe(df_temp23, 90)


# In[258]:


# We create the following Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_prepr['mths_since_last_record:Missing'] = np.where((df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_record:0-2'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
df_inputs_prepr['mths_since_last_record:3-20'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
df_inputs_prepr['mths_since_last_record:21-31'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
df_inputs_prepr['mths_since_last_record:32-80'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
df_inputs_prepr['mths_since_last_record:81-86'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
df_inputs_prepr['mths_since_last_record:>86'] = np.where((df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)


# # EXPORT THE PREPROCESS TRAIN AND TEST DATASETS

# In[259]:



path = os.getcwd()
path


# In[260]:



ls


# In[261]:


#Making a data folder
os.mkdir('data')


# In[262]:


loan_data_inputs_test = df_inputs_prepr


# In[263]:


#Exporting the preprocesse train and Test dataset as csv files
loan_data_inputs_train.to_csv(os.getcwd() + r'\data\loan_data_inputs_train.csv')
loan_data_targets_train.to_csv(os.getcwd() + r'\data\loan_data_targets_train.csv')
loan_data_inputs_test.to_csv(os.getcwd() + r'\data\loan_data_inputs_test.csv')
loan_data_targets_test.to_csv(os.getcwd() + r'\data\loan_data_targets_test.csv')

