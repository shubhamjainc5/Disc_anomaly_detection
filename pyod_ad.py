# -*- coding: utf-8 -*-
"""Example of using ECOD for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.ecod import ECOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

from joblib import dump, load
import json
import copy

from database_connector import SQLDataBase
from dateutil.relativedelta import relativedelta
import pprint

kpi_cols = ['booked_revenue', 'booked_units', 'shipped_revenue',
    'cancel_revenue', 'shipped_margin', 'shipped_units', 'cancel_units',
    'return_units', 'visits']

with open("config.json", "r") as f:
    domain_config = json.load(f)

DB_CREDS = domain_config["db_creds"]

def merge_quarter_entries(df):
    new_df = copy.deepcopy(df)
    
    remove_idx = []

    for idx,rows in new_df.iterrows():

        if pd.to_datetime(rows['week_start']).weekday()!= 0:
            remove_idx.append(idx)
            for col in kpi_cols:
                new_df.loc[idx+1,col] = new_df.loc[idx,col] + new_df.loc[idx+1,col]
            
    new_df.drop(new_df.index[remove_idx], inplace=True)
    new_df.dropna(inplace=True)

    return new_df

def _execute_sql_query(sql_db, sql_query):
    """function to execute sql query"""
    if sql_db.is_conn_closed():
        sql_db.reconnect()

    try:
        df = sql_db.execute_sql(sql_query)
        df = df.dropna().reset_index(drop=True)
        return df, True
    except Exception:
        return None, False


sql_db = SQLDataBase(DB_CREDS)


def inference_model_result():
    # result = pd.read_csv('Train_lenovo_aggdata_202312141347.csv')
    # test_df = pd.read_csv('test_lenovo_aggdata_202312141347.csv')

    sql_query = "SELECT * FROM spt_anomaly_data"
    df, _ = _execute_sql_query(sql_db, sql_query)

    merged_df = merge_quarter_entries(df)
    input_df = copy.deepcopy(merged_df)

    merged_df['week_start'] = merged_df['week_start'].apply(lambda x: pd.to_datetime(x))

    last_date = merged_df.iloc[-1,:]['week_start']
    start_date = last_date + relativedelta(months=-9)

    train_df = input_df.loc[(merged_df['week_start'] < start_date)]
    test_df = input_df.loc[(merged_df['week_start'] >= start_date)]

    x_train = train_df[kpi_cols]
    x_test = test_df[kpi_cols]

    # train ECOD detector
    clf_name = 'ECOD'
    clf = ECOD()

    # you could try parallel version as well.
    # clf = ECOD(n_jobs=2)
    clf.fit(x_train)
    # save the model
    dump(clf, 'clf.joblib')

    # load the model
    clf1 = load('clf.joblib')

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf1.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf1.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_pred_test = clf1.predict(x_test)# outlier labels (0 or 1)
    y_test_scores = clf1.decision_function(x_test)# outlier scores

    test_df.loc[:,'Anomaly_flag'] = y_pred_test
    test_df.loc[:,'Anomaly_score'] = y_test_scores

    return test_df.to_json(orient="records")



    # fig, ax = plt.subplots(figsize=(10,6))

    # #anomaly
    # a = test_df.loc[y_pred_test == 1]
    # outlier_index=list(a.index)
    # ax.plot(test_df['booked_revenue'], color='black', label = 'Normal', linewidth=1.5)
    # ax.scatter(a.index ,a['booked_revenue'], color='red', label = 'Anomaly', s=16)
    # #ax.plot(pd.Series(prediction_score_DL.flatten()*10), color='blue', label = 'Score', linewidth=0.5)


    # plt.legend()
    # plt.title("Anamoly Detection Using DeepLog")
    # plt.xlabel('Date')
    # plt.ylabel('booked_revenue')
    # plt.show()

    # print('--')
