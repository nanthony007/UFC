import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import gzip
import dill


def binarize(result):
    if result == 'W':
        return 1
    elif result == 'L':
        return 0


def dateify(date):
    return datetime.strptime(date, '%Y-%m-%d')
    

def train_model():
    """Train a machine learning model to predict UFC fight winner.

    Returns
    -------
    best_model : scikit-learn trained classifier
        Returns the best model found through tuning the hyperparameters.
        """
    cwd = os.getcwd()
    fightdata = pd.read_csv(cwd + '/UFC_stats.csv', engine='python')
    fightdata = fightdata[:int(len(fightdata)/2)]
    fightdata.dropna(subset=['Date'], inplace=True)
    fightdata.dropna(subset=['F1_DOB'], inplace=True)
    fightdata.dropna(subset=['F2_DOB'], inplace=True)
    fightdata['DateTime'] = fightdata.apply(lambda x: dateify(x.Date), axis=1)
    fightdata['F1_dob_datetime'] = fightdata.apply(lambda x: dateify(x.F1_DOB), axis=1)
    fightdata['F2_dob_datetime'] = fightdata.apply(lambda x: dateify(x.F2_DOB), axis=1)

    # drop if date before -> UFC 21?
    dt = datetime.strptime('July 16, 1999', '%B %d, %Y')
    fight_data = fightdata[fightdata.DateTime >= dt]

    # drop location, F2_result, Date
    fights = fight_data.drop(['F1_DOB', 'F2_DOB', 'F1_profile_url', 'F2_profile_url',
                             'F2_result', 'Date', 'Location', 'DateTime', 'Attendance'],
                             axis=1).dropna()

    # change F1_result to binary_result column
    fights['BinaryResult'] = fights.apply(lambda x: binarize(x.F1_result), axis=1)
    fights.dropna(subset=['BinaryResult'], inplace=True)
    # will drop binary result inplace later

    results = fights['BinaryResult'].values.tolist()  # predictor variable y
    fights = fights.drop(columns=['BinaryResult'])  # now drop

    # split into two dfs
    f1_df = fights.filter(regex='^F1')
    f2_df = fights.filter(regex='^F2')

    # get averages from matches NOT the current put into two lists
    f1_predata = []
    for i, _ in f1_df.iterrows():
        f1_predata.append(f1_df.drop(i).mean().tolist())
    print('F1_df done')
    f2_predata = []
    for i, _ in f2_df.iterrows():
        f2_predata.append(f2_df.drop(i).mean().tolist())
    print('F2_df done')
    
    # subtract lists to get diff between F1 averages and F2 averages before match
    data = np.subtract(f1_predata, f2_predata)

    rec_fights = 100
    x_train, x_test, y_train, y_test = train_test_split(data, results, shuffle=False,
                                                        test_size=rec_fights*100 / len(data) / 100,
                                                        random_state=42)

    logistic_classifier = LogisticRegressionCV(cv=5, max_iter=1000)
    logistic_classifier.fit(x_train, y_train)
    print(logistic_classifier.score(x_test, y_test))
   
    return logistic_classifier


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


def deploy_model(file_name='UFC_logistic_W-L_model_2.dill.gz'):
    """Return the loaded trained model.

    Parameters
    ----------
    file_name : str (default='UFC_logistic_WL_model_2.dill.gz')
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


if __name__ == '__main__':
    deploy_model()
