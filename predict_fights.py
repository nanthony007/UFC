import pandas as pd
import numpy as np
import os
from modeling import deploy_model


def focus_only_stats(df):
    return df[['F1_Height', 'F1_Weight', 'F1_Reach',  'F1_DOB', 'F1_KD', 'F1_SS_hit',
               'F1_SS_att', 'F1_totalStrikes_hit', 'F1_totalStrikes_att',
               'F1_TD_conv', 'F1_TD_att', 'F1_Sub', 'F1_pass', 'F1_rev',
               'F1_head_hit', 'F1_head_att', 'F1_body_conv', 'F1_body_att',
               'F1_leg_conv', 'F1_leg_att', 'F1_distance_conv', 'F1_distance_att',
               'F1_clinch_conv', 'F1_clinch_att', 'F1_ground_conv', 'F1_ground_att']]


def predict_fight():
    f1_name = input('Fighter 1: ')
    f2_name = input('Fighter 2: ')
    cwd = os.getcwd()
    df = pd.read_csv(cwd + '/UFC_stats.csv', engine='python')
    f1_df = df[df['Fighter1'] == f1_name]
    f2_df = df[df['Fighter1'] == f2_name]
    f1_less = focus_only_stats(f1_df)
    f2_less = focus_only_stats(f2_df)
    f1_ewm = f1_less.ewm(alpha=0.5).mean().iloc[[-1]].values
    f2_ewm = f2_less.ewm(alpha=0.5).mean().iloc[[-1]].values
    loaded_model = deploy_model()
    subbed = np.subtract(f1_ewm, f2_ewm)
    print('Predicted result: ', loaded_model.predict(subbed)[0])
    if loaded_model.predict(subbed)[0] == 1.0:
        print('Predicted winner: ', f1_name)
    elif loaded_model.predict(subbed)[0] == 0.0:
        print('Predicted winner: ', f2_name)
    print('Chance of ', f1_name, ' winning: ',
          str(loaded_model.predict_proba(subbed)[0, 1]*100) + '%')


if __name__ == "__main__":
    predict_fight()
