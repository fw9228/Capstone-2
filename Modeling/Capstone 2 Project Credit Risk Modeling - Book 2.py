#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split

import os


# # LOAD THE EXPORTED DATASETS

# In[2]:


#input-train data
filename1 = r'C:\Users\13134\Documents\GitHub\Capstone-2\Data Pre-Processing\data\loan_data_inputs_train.csv'
loan_data_inputs_train = pd.read_csv(filename1, index_col = 0)

#input-test data
filename2 = r'C:\Users\13134\Documents\GitHub\Capstone-2\Data Pre-Processing\data\loan_data_inputs_test.csv'
loan_data_inputs_test = pd.read_csv(filename2, index_col = 0)

#targets-train data
filename3 = r'C:\Users\13134\Documents\GitHub\Capstone-2\Data Pre-Processing\data\loan_data_targets_train.csv'
loan_data_targets_train = pd.read_csv(filename3, index_col = 0)

#targets-test data
filename4 = r'C:\Users\13134\Documents\GitHub\Capstone-2\Data Pre-Processing\data\loan_data_targets_test.csv'
loan_data_targets_test = pd.read_csv(filename4, index_col = 0)


# **Exploring the Data**

# In[3]:


#inputs-train data
loan_data_inputs_train.head()


# In[4]:


#targets-train data
loan_data_targets_train.head()


# In[5]:


#inputs-test data
loan_data_inputs_test.head()


# In[6]:


#targets-test data
loan_data_targets_test.head()


# In[7]:


loan_data_inputs_train.shape


# In[8]:


loan_data_targets_train.shape


# In[9]:


loan_data_inputs_test.shape


# In[10]:


loan_data_targets_test.shape


# **Selecting the Features**

# In[11]:


# Here we select a limited set of input variables in a new dataframe.
inputs_train_with_ref_cat = loan_data_inputs_train[['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'delinq_2yrs:0',
'delinq_2yrs:1-3',
'delinq_2yrs:>=4',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'open_acc:0',
'open_acc:1-3',
'open_acc:4-12',
'open_acc:13-17',
'open_acc:18-22',
'open_acc:23-25',
'open_acc:26-30',
'open_acc:>=31',
'pub_rec:0-2',
'pub_rec:3-4',
'pub_rec:>=5',
'total_acc:<=27',
'total_acc:28-51',
'total_acc:>=52',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31'
]]


# In[12]:


# Here we store the names of the reference category dummy variables in a list.

ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'delinq_2yrs:>=4',
'inq_last_6mths:>6',
'open_acc:0',
'pub_rec:0-2',
'total_acc:<=27',
'acc_now_delinq:0',
'total_rev_hi_lim:<=5K',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']


# In[13]:


# From the dataframe with input variables, we drop the variables with variable names in the list with reference categories. 

inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)
inputs_train.head()


# # PROPABILITY OF DEFAULT (PD) MODEL ESTIMATION

# **We will be using Logistic Regression for our PD Model**

# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn import linear_model
import scipy.stats as stat


# In[15]:


# P values for sklearn logistic regression.

# Class to display p-values for logistic regression in sklearn.

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values


# In[16]:


# We create an instance of an object from the newly created 'LogisticRegression_with_p_values()' class
reg = LogisticRegression_with_p_values()


# In[17]:





# Sets the pandas dataframe options to display all columns/ rows.
pd.options.display.max_rows = None


# In[18]:


# Estimates the coefficients of the object from the 'LogisticRegression' class
# with inputs (independent variables) contained in the first dataframe
# and targets (dependent variables) contained in the second dataframe.
reg.fit(inputs_train, loan_data_targets_train)


# In[19]:


# Displays the intercept contain in the estimated ("fitted") object from the 'LogisticRegression' class.
reg.intercept_


# In[20]:


# Displays the coefficients contained in the estimated ("fitted") object from the 'LogisticRegression' class.
reg.coef_


# In[21]:


# Stores the names of the columns of a dataframe in a variable.
feature_name = inputs_train.columns.values


# In[22]:


# Creates a dataframe with a column titled 'Feature name' and row values contained in the 'feature_name' variable.
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)


# Creates a new column in the dataframe, called 'Coefficients',
# with row values the transposed coefficients from the 'LogisticRegression' object.
summary_table['Coefficients'] = np.transpose(reg.coef_)


# Increases the index of every row of the dataframe with 1.
summary_table.index = summary_table.index + 1

# Assigns values of the row with index 0 of the dataframe.
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# Sorts the dataframe by index.
summary_table = summary_table.sort_index()

#Print Summary table
summary_table


# In[23]:


# This is a list.
# We take the result of the newly added method 'p_values' and store it in a variable 'p_values'
p_values = reg.p_values


# In[24]:


# Add the intercept for completeness.
# We add the value 'NaN' in the beginning of the variable with p-values since intercept has no p-value.
p_values = np.append(np.nan, np.array(p_values))


# In[25]:


# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable
summary_table['p_values'] = p_values


# In[26]:


#Print table
summary_table


# Each orginal independent variable is represented by several dummy variables. Therefore, if one of few dummy variables representing one original independent variable are statistically significant, it would be best to retain all dummy variables that represent that original independent variable.

# **Conventionally, if a p-value is lower than 0.05, we conclude that the coefficient of a variable is statistically significant.**
# 
# In the next phase, we are going to remove some features, the coefficients for all or almost all of the dummy variables for which are not statistically significant (greater than 0.1). The independent variables which p-values are not statistically significant: `open_acc` and `pub_rec`. We will remove them and reconstruct our PD model again (Final PD Model).
# 

# **Final PD Model**

# In[27]:


# Open_acc and pub_rec Variables are removed
inputs_train_with_ref_cat = loan_data_inputs_train[['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'delinq_2yrs:0',
'delinq_2yrs:1-3',
'delinq_2yrs:>=4',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'total_acc:<=27',
'total_acc:28-51',
'total_acc:>=52',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31'
]]


# In[28]:


# Open_acc and pub_rec Variables are removed

ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'delinq_2yrs:>=4',
'inq_last_6mths:>6',
'total_acc:<=27',
'acc_now_delinq:0',
'total_rev_hi_lim:<=5K',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']


# In[29]:


inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)
inputs_train.head()


# In[30]:


# Here we run a new model.
reg2 = LogisticRegression_with_p_values()
reg2.fit(inputs_train, loan_data_targets_train)


# In[31]:


feature_name = inputs_train.columns.values


# In[32]:


# Same as above.
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg2.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg2.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# In[33]:


# We add the 'p_values' here, just as we did before.
p_values = reg2.p_values
p_values = np.append(np.nan,np.array(p_values))
summary_table['p_values'] = p_values
summary_table
# Here we get the results for our final PD model.


# **Interpreting the coefficients**: High coefficients means chances of being a good borrower and thus lower default and vise versa.
# 
# For example, lets take the `grade` variable, G had a lower weight of evidence(woe) and thus we made it the reference category. This makes it automatically to have a coefficient of zero
# 
# Now let's see what the odds are for someone with a grade D (0.829629) to be better than someone with a grade G (0):
# `In(odds(grade=D)/odds(grade=G))=e^(0.829629 - 0)`. This is equal to 2.2924 as e value is 2.71828. So the odds of someone with a grade D to be a good borrower are 2.2924 times higher than the odds of someone with a grade G.
# 
# Another example, let's see what the odds are for someone with a grade B (1.621832) to be better than someone with a grade D (0.829629): `In(odds(grade=B)/odds(grade=D))= e^(1.621832 - 0.829629)`. This is equal to 2.21. So the odds of someone with a grade B to be a good borrower are 2.21 times higher than the odds of someone with a grade D.
# 
# Note that direct comparisons are possible only between categories coming from one and the same original independent variable. Thus, for instance we cannot compare one dummy variable `grade` coefficient to that of `home_ownership` dummy variable.

# **Saving our PD Model**

# In[35]:


import pickle


# Here we export our model to a 'SAV' file with file name 'pd_model.sav'.
pickle.dump(reg2, open('pd_model.sav', 'wb'))


# # PD MODEL VALIDATION (Using the Test dataset)

# **Out-of-sample validation**

# In[40]:


# Here, from the dataframe with inputs for testing, we keep the same variables that we used in our final PD model.
# Open_acc and pub_rec Variables are removed
inputs_test_with_ref_cat = loan_data_inputs_test[['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'delinq_2yrs:0',
'delinq_2yrs:1-3',
'delinq_2yrs:>=4',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'total_acc:<=27',
'total_acc:28-51',
'total_acc:>=52',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31'
]]


# In[41]:


# And here, in the list below, we keep the variable names for the reference categories,
# only for the variables we used in our final PD model.
# Open_acc and pub_rec Variables are removed

ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'delinq_2yrs:>=4',
'inq_last_6mths:>6',
'total_acc:<=27',
'acc_now_delinq:0',
'total_rev_hi_lim:<=5K',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']


# In[42]:


inputs_test = inputs_test_with_ref_cat.drop(ref_categories, axis = 1)


# In[43]:


# Calculates the predicted values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
y_hat_test = reg2.model.predict(inputs_test)


# In[44]:


# This is an array of predicted discrete classess (in this case, 0s and 1s).
y_hat_test


# In[45]:


# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
y_hat_test_proba = reg2.model.predict_proba(inputs_test)


# In[46]:


# This is an array of arrays of predicted class probabilities for all classes.
# In this case, the first value of every sub-array is the probability for the observation to belong to the first class, i.e. 0,
# and the second value is the probability for the observation to belong to the first class, i.e. 1.
#The estimated probabilities are categorized into being good or bad by applying a cut-off point of 0.5 (50%):
#Estimated probability of <=50%, is classified as bad (0)
#Estimated probability of >50%, is classified as good (1)

y_hat_test_proba


# In[47]:


# Here we take all the arrays in the array, and from each array, we take all rows, and only the element with index 1,
# that is, the second element.
# In other words, we take only the probabilities for being 1.
y_hat_test_proba[:][:,1]


# In[48]:


# We store these probabilities in a new variable.
y_hat_test_proba = y_hat_test_proba[: ][: , 1]


# In[49]:


# This variable contains an array of probabilities of being 1.
y_hat_test_proba


# In[50]:


#We will be using the target_test dataset
loan_data_targets_test_temp = loan_data_targets_test


# In[51]:


# We reset the index of a dataframe.
loan_data_targets_test_temp.reset_index(drop = True, inplace = True)


# In[52]:


# Concatenates two dataframes.
df_actual_predicted_probs = pd.concat([loan_data_targets_test_temp, pd.DataFrame(y_hat_test_proba)], axis = 1)


# In[53]:


#df_actual_predicted_probs


# In[54]:


#checking the shape of the dataset
df_actual_predicted_probs.shape


# In[55]:


#Renaming the columns
df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba']


# In[56]:


# Makes the index of one dataframe equal to the index of another dataframe.
df_actual_predicted_probs.index = loan_data_inputs_test.index


# In[57]:


df_actual_predicted_probs.head()


# **Accuracy and Area under the Curve (AUC)**

# In[58]:


# We create a new column with an indicator,
# where every observation that has predicted probability greater than the threshold has a value of 1,
# and every observation that has predicted probability lower than the threshold has a value of 0.
#Here lets set our treshold to be 0.5
tr = 0.5
df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)


# In[59]:


# We Create a cross-table where the actual values are displayed by rows and the predicted values by columns.
# This table is known as a Confusion Matrix.
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted'])


# In[60]:


#We can convert them into percentages
# Here we divide each value of the table by the total number of observations,
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]


# In[61]:


# Here we calculate Accuracy of the model, which is the sum of the diagonal rates.

(pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]


# Accuracy is the sum of true positives (394757) and true negatives (319), divided by the total number of obervations (adding up all 4). Our model accuracy rate is 87%. Although the accuracy rate is high, we cannot conclude the model is best. We have to do further analysis.
# 
# From all observations that are actually good, 394757 are predicted to be good and only 312 are predicted to be bad. 
# 
# However, from observations that are actually bad, only 319 are correctly predicted as bad while 56746 are predicted to be good. This is a problem as it turns out that under a treshold of 0.5, the model generates alot of false positive observations (56746).
# 
# If we grant loans based on this model under the threshold of 0.5, it means a lot of bad applicants will be given loans given our high false positive numbers.
# 
# Therefore we have to set a more conservative threshold.

# In[62]:


#Setting another threshold of 0.9
tr = 0.9
df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)


# In[63]:


# We Create a cross-table where the actual values are displayed by rows and the predicted values by columns.
# This table is known as a Confusion Matrix.
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted'])


# In[64]:


#We can convert them into percentages
# Here we divide each value of the table by the total number of observations,
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]


# In[65]:


# Here we calculate Accuracy of the model, which is the sum of the diagonal rates.
(pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]


# With our new threshold of 0.9, the accuracy rate has reduce to 56%. However looking at the confusion matrix, the actual bad obervations of 57065 (46644 + 10421), our model correctly predicts 46644 as bad and only 10421 as good (false positives). Thus the false positives is lower compared to the first model with a threshold of 0.5.
# 
# Also, for 395059 (188294 + 206775) observations that are actually good, our model predicted 206775 (true positives) to be good and only 188294 to be bad.
# 
# Therefore if we use this model to grant loans under the 0.9 threshold, they will reduce the nunber of defaults dramatically and also the number of overall approved applications.
# 
# Furthermore, a 0.9 threshold may be too conservative and may lead to lose of customers who may have wanted loans, and possible lose of business. With credit risk modeling, we want to reduce risk but also still want to give out loans so as to stay in business as a bank/financial institution. Thus, the overall accuracy is not the universal measure for a Probability of Default (PD) Model. The rate of the true positives and false positives are far more important than the overall accuracy rate.

# **Plotting Receiver Operating Characteristic (ROC) Curve**

# In[66]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[67]:


# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])


# In[68]:


# Here we store each of the three arrays in a separate variable. 
fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])


# In[69]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[70]:


# We plot the false positive rate along the x-axis and the true positive rate along the y-axis,
# thus plotting the ROC curve.
plt.plot(fpr, tpr)

# We plot a seconary diagonal line, with dashed line style and black color.
plt.plot(fpr, fpr, linestyle = '--', color = 'k')

# We name the x-axis "False positive rate".
plt.xlabel('False positive rate')

# We name the x-axis "True positive rate".
plt.ylabel('True positive rate')

# We name the graph "ROC curve".
plt.title('ROC curve')


# In[71]:


# Calculating the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities

AUROC = roc_auc_score(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])
AUROC


# The commond interpretation for area under the curve results is that:
# `The moodel is bad: 50 -60%`
# `The model is Poor: 60 - 70%`
# `The model is Fair: 70 - 80%`
# `The model is Good: 80 - 90%`
# `The model is Excellent: 90 - 100%`
# 
# We can conclude our model is fair.

# **Gini and Kolmogorov-Smirnov**
# 
# These two are widely accepted in the credit risk modeling community for evaluation of model performance

# In[72]:


# Sorts a dataframe by the values of a specific column.
df_actual_predicted_probs = df_actual_predicted_probs.sort_values('y_hat_test_proba')


# In[73]:


# We reset the index of a dataframe and overwrite it.
df_actual_predicted_probs = df_actual_predicted_probs.reset_index()


# In[74]:


# We calculate the cumulative number of all observations.
# We use the new index for that. Since indexing in ython starts from 0, we add 1 to each index.
df_actual_predicted_probs['Cumulative N Population'] = df_actual_predicted_probs.index + 1

# We calculate cumulative number of 'good' borrowers, which is the cumulative sum of the column with actual observations.
df_actual_predicted_probs['Cumulative N Good'] = df_actual_predicted_probs['loan_data_targets_test'].cumsum()


# We calculate cumulative number of 'bad' borrwers, which is
# the difference between the cumulative number of all observations and cumulative number of 'good' for each row.
df_actual_predicted_probs['Cumulative N Bad'] = df_actual_predicted_probs['Cumulative N Population'] - df_actual_predicted_probs['loan_data_targets_test'].cumsum()


# In[75]:


# We calculate the cumulative percentage of all observations.
df_actual_predicted_probs['Cumulative Perc Population'] = df_actual_predicted_probs['Cumulative N Population'] / (df_actual_predicted_probs.shape[0])

# We calculate cumulative percentage of 'good'.
df_actual_predicted_probs['Cumulative Perc Good'] = df_actual_predicted_probs['Cumulative N Good'] / df_actual_predicted_probs['loan_data_targets_test'].sum()

# We calculate the cumulative percentage of 'bad'.
df_actual_predicted_probs['Cumulative Perc Bad'] = df_actual_predicted_probs['Cumulative N Bad'] / (df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['loan_data_targets_test'].sum())


# In[76]:


# Plot Gini
# We plot the cumulative percentage of all along the x-axis and the cumulative percentage 'good' along the y-axis,
# thus plotting the Gini curve.
plt.plot(df_actual_predicted_probs['Cumulative Perc Population'], df_actual_predicted_probs['Cumulative Perc Bad'])


# We plot a seconary diagonal line, with dashed line style and black color.
plt.plot(df_actual_predicted_probs['Cumulative Perc Population'], df_actual_predicted_probs['Cumulative Perc Population'], linestyle = '--', color = 'k')


# We name the x-axis "Cumulative % Population".
plt.xlabel('Cumulative % Population')


# We name the y-axis "Cumulative % Bad".
plt.ylabel('Cumulative % Bad')


# We name the graph "Gini".
plt.title('Gini')


# In[77]:


# Here we calculate Gini from area under the curve
Gini = AUROC * 2 - 1
Gini


# In[79]:


# Plotting Kolmogorov-Smirnov (KS)
# We plot the predicted (estimated) probabilities along the x-axis and the cumulative percentage 'bad' along the y-axis,
# colored in red.
plt.plot(df_actual_predicted_probs['y_hat_test_proba'], df_actual_predicted_probs['Cumulative Perc Bad'], color = 'r')


# We plot the predicted (estimated) probabilities along the x-axis and the cumulative percentage 'good' along the y-axis,
# colored in blue.
plt.plot(df_actual_predicted_probs['y_hat_test_proba'], df_actual_predicted_probs['Cumulative Perc Good'], color = 'b')


# We name the x-axis "Estimated Probability for being Good".
plt.xlabel('Estimated Probability for being Good')


# We name the y-axis "Cumulative %".
plt.ylabel('Cumulative %')

# We name the graph "Kolmogorov-Smirnov".
plt.title('Kolmogorov-Smirnov')


# We focus on the maximum distance between the red and the blue curve

# In[80]:


# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted_probs['Cumulative Perc Bad'] - df_actual_predicted_probs['Cumulative Perc Good'])
KS


# From the results of Gini and KS, the two cumulative distribution functions are sufficiently far away from each other and our model has satisfactory predictive power.

# # APPLYING OUR PD MODEL FOR DECISION MAKING

# **Calculating PD of individual accounts**

# In[81]:


# Sets the pandas dataframe options to display all columns/ rows.
pd.options.display.max_columns = None


# In[82]:


inputs_test_with_ref_cat.head()


# In[83]:


summary_table


# In[84]:


y_hat_test_proba


# **Creating a Scorecard**
# 
# Applying the scorecard is just like applying the PD model itself. The scorecard produces an individual credit worthiness assessment that directly corresponds to a specific probability of default, and because these credit worthiness are named after scorecard, they are called credit scores.

# In[85]:


summary_table


# In[86]:


ref_categories


# In[87]:


# We create a new dataframe with one column. Its values are the values from the 'reference_categories' list.
# We name it 'Feature name'.
df_ref_categories = pd.DataFrame(ref_categories, columns = ['Feature name'])

# We create a second column, called 'Coefficients', which contains only 0 values.
df_ref_categories['Coefficients'] = 0

# We create a third column, called 'p_values', with contains only NaN values.
df_ref_categories['p_values'] = np.nan

df_ref_categories


# In[88]:


# Concatenates two dataframes.
df_scorecard = pd.concat([summary_table, df_ref_categories])


# We reset the index of a dataframe.
df_scorecard = df_scorecard.reset_index()

df_scorecard


# In[89]:


# We create a new column, called 'Original feature name', which contains the value of the 'Feature name' column,
# up to the column symbol.
df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]


# In[90]:


df_scorecard


# In[91]:


min_score = 300
max_score = 850


# In[92]:


# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their minimum.
df_scorecard.groupby('Original feature name')['Coefficients'].min()


# In[93]:


# Up to the 'min()' method everything is the same as in te line above.
# Then, we aggregate further and sum all the minimum values.
min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
min_sum_coef


# In[94]:


# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their maximum.
df_scorecard.groupby('Original feature name')['Coefficients'].max()


# In[95]:


# Up to the 'min()' method everything is the same as in te line above.
# Then, we aggregate further and sum all the maximum values.
max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
max_sum_coef


# In[96]:


# We multiply the value of the 'Coefficients' column by the ration of the differences between
# maximum score and minimum score and maximum sum of coefficients and minimum sum of cefficients.
df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
df_scorecard


# In[97]:


# We divide the difference of the value of the 'Coefficients' column and the minimum sum of coefficients by
# the difference of the maximum sum of coefficients and the minimum sum of coefficients.
# Then, we multiply that by the difference between the maximum score and the minimum score.
# Then, we add minimum score. 

df_scorecard['Score - Calculation'][0] = ((df_scorecard['Coefficients'][0] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score

df_scorecard


# In[98]:


# We round the values of the 'Score - Calculation' column.
df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()

df_scorecard


# In[99]:


# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their minimum.
# Sums all minimum values.
min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()
min_sum_score_prel


# In[100]:


# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their maximum.
# Sums all maximum values.
max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()
max_sum_score_prel


# In[101]:


# One (1) has to be subtracted from the maximum score for one original variable. 
# We will determine which one by evaluating based on differences.
df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']
df_scorecard


# In[105]:


#The highest score for the Difference (0.493807) calculated above is index at number 80
#At the index of 80, the 'score-calculation' value is 12.506193
# We will use those values and not the rounded value of 13
df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
df_scorecard['Score - Final'][80] = 12
df_scorecard


# In[106]:


# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their minimum.
# Sums all minimum values.
min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Final'].min().sum()
min_sum_score_prel


# In[107]:


# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their maximum.
# Sums all maximum values.
max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Final'].max().sum()
max_sum_score_prel


# **Caclulating Credit Score**

# In[108]:


inputs_test_with_ref_cat.head()


# In[109]:


df_scorecard


# In[110]:


inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat


# In[111]:


# We insert a column in the dataframe, with an index of 0, that is, in the beginning of the dataframe.
# The name of that column is 'Intercept', and its values are 1s.
inputs_test_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)


# In[112]:


inputs_test_with_ref_cat_w_intercept.head()


# In[113]:


# Here, from the 'inputs_test_with_ref_cat_w_intercept' dataframe, we keep only the columns with column names,
# exactly equal to the row values of the 'Feature name' column from the 'df_scorecard' dataframe.
inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat_w_intercept[df_scorecard['Feature name'].values]


# In[114]:


inputs_test_with_ref_cat_w_intercept.head()


# In[115]:


scorecard_scores = df_scorecard['Score - Final']


# In[116]:


inputs_test_with_ref_cat_w_intercept.shape


# In[117]:


scorecard_scores.shape


# In[118]:


# we will reshape the scorecard_scores
scorecard_scores = scorecard_scores.values.reshape(113, 1)


# In[119]:


scorecard_scores.shape


# In[120]:


# Here we multiply the values of each row of the dataframe by the values of each column of the variable,
# which is an argument of the 'dot' method, and sum them. It's essentially the sum of the products.
y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)


# In[121]:


y_scores.head()


# We have successfully calculated the credit scores of all borrowers from the test data

# # FROM CREDIT SCORE TO PROBABILITY OF DEFAULT (PD)
# 

# **We were able to calculate the credit score of borrowers from the PD Model. We can also do vice-versa**

# In[122]:


# We divide the difference between the scores and the minimum score by
# the difference between the maximum score and the minimum score.
# Then, we multiply that by the difference between the maximum sum of coefficients and the minimum sum of coefficients.
# Then, we add the minimum sum of coefficients.
sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef


# In[123]:


# Here we divide an exponent raised to sum of coefficients from score by
# an exponent raised to sum of coefficients from score plus one.
y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)
y_hat_proba_from_score.head()


# In[124]:


#See the results from our previous PD model below
# Approximately the same probability we got as above using the credit score approach.
y_hat_test_proba[0: 5]


# **Setting Cut-offs**
# 
# Using our previous threshold of 0.9 under the confusion matrix and ROC Curve (with the same codes), we can calculate a cut-off point. This will be the point to consider, to qualify borrowers for loans

# In[125]:


# We need the confusion matrix again.
#np.where(np.squeeze(np.array(loan_data_targets_test)) == np.where(y_hat_test_proba >= tr, 1, 0), 1, 0).sum() / loan_data_targets_test.shape[0]
tr = 0.9
df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)
#df_actual_predicted_probs['loan_data_targets_test'] == np.where(df_actual_predicted_probs['y_hat_test_proba'] >= tr, 1, 0)


# In[126]:


pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted'])


# In[127]:


pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]


# In[128]:


(pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]


# In[129]:


roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])


# In[130]:


fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])


# In[131]:


plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')


# In[132]:


thresholds


# In[133]:


thresholds.shape


# In[134]:


# We concatenate 3 dataframes along the columns.
df_cutoffs = pd.concat([pd.DataFrame(thresholds), pd.DataFrame(fpr), pd.DataFrame(tpr)], axis = 1)


# In[135]:


# We name the columns of the dataframe 'thresholds', 'fpr', and 'tpr'
df_cutoffs.columns = ['thresholds', 'fpr', 'tpr']


# In[136]:


df_cutoffs.head()


# In[137]:


# Let the first threshold (the value of the thresholds column with index 0) be equal to a number, very close to 1
# but smaller than 1, say 1 - 1 / 10 ^ 16.
df_cutoffs['thresholds'][0] = 1 - 1 / np.power(10, 16)


# In[138]:


# The score corresponsing to each threshold equals:
# The the difference between the natural logarithm of the ratio of the threshold and 1 minus the threshold and
# the minimum sum of coefficients
# multiplied by
# the sum of the minimum score and the ratio of the difference between the maximum score and minimum score and 
# the difference between the maximum sum of coefficients and the minimum sum of coefficients.
df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) - min_sum_coef) * ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round()


# In[139]:


df_cutoffs.head()


# In[140]:


df_cutoffs['Score'][0] = max_score


# In[141]:


df_cutoffs.head()


# In[142]:


df_cutoffs.tail()


# In[143]:


# We define a function called 'n_approved' which assigns a value of 1 if a predicted probability
# is greater than the parameter p, which is a threshold, and a value of 0, if it is not.
# Then it sums the column.
# Thus, if given any percentage values, the function will return
# the number of rows wih estimated probabilites greater than the threshold. 
def n_approved(p):
    return np.where(df_actual_predicted_probs['y_hat_test_proba'] >= p, 1, 0).sum()


# In[144]:


# Assuming that all credit applications above a given probability of being 'good' will be approved,
# when we apply the 'n_approved' function to a threshold, it will return the number of approved applications.
# Thus, here we calculate the number of approved appliations for al thresholds.
df_cutoffs['N Approved'] = df_cutoffs['thresholds'].apply(n_approved)


# Then, we calculate the number of rejected applications for each threshold.
# It is the difference between the total number of applications and the approved applications for that threshold.
df_cutoffs['N Rejected'] = df_actual_predicted_probs['y_hat_test_proba'].shape[0] - df_cutoffs['N Approved']


# Approval rate equalts the ratio of the approved applications and all applications.
df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / df_actual_predicted_probs['y_hat_test_proba'].shape[0]


# Rejection rate equals one minus approval rate.
df_cutoffs['Rejection Rate'] = 1 - df_cutoffs['Approval Rate']


# In[145]:


df_cutoffs.head()


# In[146]:


# Here we display the dataframe with cutoffs form line with index 5000 to line with index 6200.
df_cutoffs.iloc[5000: 6200, ]


# In[147]:


# Here we display the dataframe with cutoffs form line with index 1000 to line with index 2000.
df_cutoffs.iloc[1000: 2000, ]


# **Exporting the Model Results**

# In[148]:


#Making a Results folder
os.mkdir('Results')


# In[149]:


inputs_train_with_ref_cat.to_csv(os.getcwd() + r'\Results\inputs_train_with_ref_cat.csv')
df_scorecard.to_csv(os.getcwd() + r'\Results\df_scorecard.csv')


# In[ ]:




