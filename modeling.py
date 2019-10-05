# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFECV
import gzip
import dill
import math


#%%
def serialize_model(file_name):
    """Serialize the trained machine learning model.

    Parameters
    ----------
    file_name : str (default='____.dill')
        File name to use when persisting trained model.
    """""

    model = train_model()
    with gzip.open(file_name, 'wb') as f:
        dill.dump(model, f)


#%%
def binarize(result):
    if result == 'W':
        return 1
    elif result == 'L':
        return 0


#%%
def dateify(date):
    return datetime.strptime(date, '%Y-%m-%d')


#%%
dateify('1990-12-20')


#%%
def train_model():
    """Train a machine learning model to predict UFC fight winner.

    Returns
    -------
    best_model : scikit-learn trained classifier
        Returns the best model found through tuning the hyperparameters.
        """
    cwd = os.getcwd()
    fightdata = pd.read_csv(cwd + '/UFCstats.csv', engine='python')
    fightdata = fightdata[:int(len(fightdata)/2)]
    fightdata.dropna(subset=['Date'], inplace=True)
    fightdata.dropna(subset=['F1_DOB'], inplace=True)
    fightdata.dropna(subset=['F2_DOB'], inplace=True)
    fightdata['DateTime'] = fightdata.apply(lambda row: dateify(row.Date), axis=1)
    fightdata['F1_dob_datetime'] = fightdata.apply(lambda row: dateify(row.F1_DOB), axis=1)
    fightdata['F2_dob_datetime'] = fightdata.apply(lambda row: dateify(row.F2_DOB), axis=1)

    # drop if date before -> UFC 21?
    dt = datetime.strptime('July 16, 1999', '%B %d, %Y')
    fightdata = fightdata[fightdata.DateTime >= dt]

    # drop location, F2_result, Date
    fights = fightdata.drop(['F1_DOB', 'F2_DOB', 'F1_profile_url', 'F2_profile_url',
                             'F2_result', 'Date', 'Location', 'DateTime', 
                             'Attendance'], axis=1).dropna()

    # change F1_result to binary_result column
    fights['BinaryResult'] = fights.apply(lambda row: binarize(row.F1_result), axis=1)
    fights.dropna(subset=['BinaryResult'], inplace=True)
    # will drop binary result inplace later

    RESULTS = fights['BinaryResult'].values.tolist() # predictor variable y
    fights = fights.drop(columns=['BinaryResult']) # now drop

    # split into two dfs
    f1_df = fights.filter(regex='^F1')
    f2_df = fights.filter(regex='^F2')

    # get averages from matches NOT the current put into two lists
    f1_predata = []
    for i, row in f1_df.iterrows():
        f1_predata.append(f1_df.drop(i).mean().tolist())
    print('F1_df done')
    f2_predata = []
    for i, row in f2_df.iterrows():
        f2_predata.append(f2_df.drop(i).mean().tolist())
    print('F2_df done')
    
    # subtract lists to get diff between F1 averages and F2 averages before match
    DATA = np.subtract(f1_predata, f2_predata)

    X_train, X_test, y_train, y_test = train_test_split(DATA, RESULTS, test_size=0.20, random_state=42)

    logistic_classifier = LogisticRegressionCV(cv=5, max_iter=1000)
    logistic_classifier.fit(X_train, y_train)
    print(logistic_classifier.score(X_test, y_test))
   
    return logistic_classifier


#%%
def deploy_model(file_name='UFC_logistic_W-L_model.dill.gz'):
    """Return the loaded trained model.

    Parameters
    ----------
    file_name : str (default='UFC_logistic_WL_model.dill.gz')
        File name to use when persisting trained model.

    Returns
    -------
    model : scikit-learn trained classifier
        Returns the serialized trained model.
    """

    # if the model has not been persisted, create it
    try:
        with gzip.open(file_name, 'rb') as f:
            model = dill.load(f)
    except FileNotFoundError:
        print("Trained model not found, creating the file.")
        serialize_model(file_name)
        return deploy_model(file_name=file_name)
    
    return model


#%%
def focus_only_stats(df):
    return df[['F1_Height', 'F1_Weight', 'F1_Reach',  'F1_DOB', 'F1_KD', 'F1_SS_hit',
       'F1_SS_att', 'F1_totalStrikes_hit', 'F1_totalStrikes_att',
       'F1_TD_conv', 'F1_TD_att', 'F1_Sub', 'F1_pass', 'F1_rev',
       'F1_head_hit', 'F1_head_att', 'F1_body_conv', 'F1_body_att',
       'F1_leg_conv', 'F1_leg_att', 'F1_distance_conv', 'F1_distance_att', 
       'F1_clinch_conv', 'F1_clinch_att', 'F1_ground_conv', 'F1_ground_att']]


#%%
def get_preds():
    f1_name = input('Fighter 1: ')
    f2_name = input('Fighter 2: ')
    cwd = os.getcwd()
    df = pd.read_csv(cwd + '/UFCstats.csv', engine='python')
    f1_df = df[df['Fighter1'] == f1_name]
    f2_df = df[df['Fighter1'] == f2_name]
    f1_less = focus_only_stats(f1_df)
    f2_less = focus_only_stats(f2_df)
    f1_ewm = f1_less.ewm(alpha=0.5).mean().iloc[[-1]].values
    f2_ewm = f2_less.ewm(alpha=0.5).mean().iloc[[-1]].values
    f1_ewm[0][3] = dateify(f1_ewm[0][3])
    f2_ewm[0][3] = dateify(f2_ewm[0][3])
    loaded_model = deploy_model()
    subbed = np.subtract(f1_ewm, f2_ewm)
    print(loaded_model.predict(subbed))
#     if loaded_model.predict(subbed)[0] == 1.0:
#         print('Predicted winner: ', f1_name)
#     elif loaded_model.predict(subbed)[0] == 0.0:
#         print('Predicted winner: ', f2_name)
#     #print('Predicted result: ', loaded_model.predict(subbed)[0])
#     #print('Chance of ', f1_name, ' winning: ', 
#          # str(loaded_model.predict_proba(subbed)[0,1]*100) + '%')


#%%
get_preds()


#%%
'2018-08-10' - '2018-04-12'


#%%



#%%
df = pd.read_csv('UFCstats.csv')


#%%
df[df.Fighter1.startswith('Khalil')]


#%%
df['F1_Reach'].isna()


#%%



#%%
count = 0
for i, row in df.iterrows():
    if math.isnan(row['F1_Reach']):
        df.loc[i, 'F1_Reach'] = df.loc[i, 'F1_Height']
        count += 1
        
print(count)


#%%
len(df[df['F1_Reach'].isna()])


#%%
df[df['F1_Reach'].isna()]


#%%
df.to_csv('UFCstats.csv', index=False)


#%%



