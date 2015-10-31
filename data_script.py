import pandas as pd  # standard naming convention
import requests  # library for easy HTTP requests
import io  # library for handling file-like objects
import numpy as np  # standard naming convention
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Get a whitelisted copy of Chronic Conditions PUF data
response = requests.get('https://s3.amazonaws.com/lightstrike/workshops/mldjango/2010_Chronic_Conditions_PUF.csv')

# transform byte-data into a file-like object pandas can read
csv_file = io.StringIO(response.content.decode())

# load data into a pandas dataframe, commonly named df
df_chronic = pd.read_csv(csv_file)

# make a copy of data to build model from
model_data = df_chronic.copy()

# define constants
model_data['constant'] = np.ones(len(model_data))

# check length of dataset, should be 22003
print('length of dataset', len(model_data))

# create dummy variables
sex_dummies = pd.get_dummies(model_data['BENE_SEX_IDENT_CD'], prefix = 'sex')
age_dummies = pd.get_dummies(model_data['BENE_AGE_CAT_CD'], prefix = 'age')

# add dummy variables to model data
model_data = pd.concat([model_data, sex_dummies, age_dummies], axis=1)

# check length of dataset again, should still be 22003
print('data after concatenating age and sex dummies', len(model_data))

# define regression features
features = ['sex_1', # male effect
            'age_1',
            'age_2',
            'age_3',
            'age_4',
            'age_5',
            'age_6',
            'CC_CANCER',
            'CC_2_OR_MORE']

# define regression target
target = 'AVE_PA_PAY_PA_EQ_12'

# drop rows with null data
model_data = model_data[features + [target]].dropna()

# rename indices
model_data.reset_index(inplace = True)

# check length once more, should now be 19367
print('length of data after dropping nans from features and target', len(model_data))

# create OLS linear regression
lm_model = sm.OLS(model_data[target], model_data[features]).fit()

# view summary
print(lm_model.summary())

# it can also be viewed in HTML
lm_model.summary().as_html()