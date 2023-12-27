# -*- coding: utf-8 -*-
"""Example of using ECOD for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from logging_handler import Logger
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.ecod import ECOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

from openai_processor import generate_reasoning

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


def prepare_test_json(test_df, llm_op):
    test_df = copy.deepcopy(test_df)
    llm_anomaly_dt = [ op['anomaly_date'] for op in llm_op]
    rows_list = []
    for idx, row in test_df.iterrows():
        value_dict ={}
        for col in kpi_cols:
            value_dict[col] = {}
            value_dict[col]['value']= row[col]
            value_dict[col]['upperBound']= row['ub_'+col]
            value_dict[col]['lowerBound']= row['lb_'+col]
            value_dict[col]['smaBound']= row['avg_'+col]

        match_idx = llm_anomaly_dt.index(row["week_start"]) if row["week_start"] in llm_anomaly_dt else -1
        narr = '' if match_idx==-1 else llm_op[match_idx]['reason']

        dict_row = {
            "week_start":row["week_start"],
            "kpi_values": value_dict,
            "narrative":narr,
            "narrativeHtml":"",
            "Anomaly_flag":row["Anomaly_flag"],
            "Anomaly_score":row["Anomaly_score"],
        }

        rows_list.append(dict_row)
    
    return rows_list


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

    narrative_df = copy.deepcopy(merged_df)

    window_num = 12
    for col in kpi_cols:
        avg_kpi = merged_df[col].rolling(window=window_num).mean().dropna()
        dev_kpi = merged_df[col].rolling(window=window_num).std().dropna()
        merged_df['avg_'+col] = avg_kpi
        merged_df['ub_'+col] = avg_kpi + dev_kpi
        merged_df['lb_'+col] = avg_kpi - dev_kpi
        merged_df['dev_'+col] = dev_kpi

    input_df = copy.deepcopy(merged_df)

    merged_df['week_start'] = merged_df['week_start'].apply(lambda x: pd.to_datetime(x))

    last_date = merged_df.iloc[-1,:]['week_start']
    start_date = last_date + relativedelta(months=-9)

    train_df = input_df.loc[(merged_df['week_start'] < start_date)]
    test_df = input_df.loc[(merged_df['week_start'] >= start_date)]
    narrative_df = narrative_df.loc[(merged_df['week_start'] >= start_date)]

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
    # y_train_pred = clf1.labels_  # binary labels (0: inliers, 1: outliers)
    # y_train_scores = clf1.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_pred_test = clf1.predict(x_test)# outlier labels (0 or 1)
    y_test_scores = clf1.decision_function(x_test)# outlier scores

    test_df.loc[:,'Anomaly_flag'] = y_pred_test
    test_df.loc[:,'Anomaly_score'] = y_test_scores

    cnt_anomaly = y_pred_test.sum()

    if cnt_anomaly>0:
        op ,api_cnt, api_tokens = generate_reasoning(narrative_df.to_csv())
        Logger.info(f"{api_cnt} API Calls were made with an average of {api_tokens} tokens per call for narrative generation")
        print(op)
    else:
        op = []

    # fig, ax = plt.subplots(10, figsize=(10,6), sharex=True)

    # #anomaly
    # a = test_df.loc[y_pred_test == 1]
    # outlier_index=list(a.index)

    # ax[0].plot(test_df['booked_revenue'], color='black', label = 'ghfjgvgv', linewidth=1.5)
    # ax[0].scatter(a.index ,a['booked_revenue'], color='red', label = 'Anomaly', s=16)

    # ax[1].plot(test_df['booked_units'], color='black', label = 'Norjhbkmbmnbmal', linewidth=1.5)
    # ax[1].scatter(a.index ,a['booked_units'], color='red', label = 'Anomaly', s=16)

    # ax[2].plot(test_df['shipped_revenue'], color='black', label = 'Normal', linewidth=1.5)
    # ax[2].scatter(a.index ,a['shipped_revenue'], color='red', label = 'Anomaly', s=16)

    # ax[3].plot(test_df['cancel_revenue'], color='black', label = 'Normal', linewidth=1.5)
    # ax[3].scatter(a.index ,a['cancel_revenue'], color='red', label = 'Anomaly', s=16)

    # ax[4].plot(test_df['shipped_margin'], color='black', label = 'Normal', linewidth=1.5)
    # ax[4].scatter(a.index ,a['shipped_margin'], color='red', label = 'Anomaly', s=16)

    # ax[5].plot(test_df['shipped_units'], color='black', label = 'Normal', linewidth=1.5)
    # ax[5].scatter(a.index ,a['shipped_units'], color='red', label = 'Anomaly', s=16)

    # ax[6].plot(test_df['cancel_units'], color='black', label = 'Normal', linewidth=1.5)
    # ax[6].scatter(a.index ,a['cancel_units'], color='red', label = 'Anomaly', s=16)

    # ax[7].plot(test_df['return_units'], color='black', label = 'Normal', linewidth=1.5)
    # ax[7].scatter(a.index ,a['return_units'], color='red', label = 'Anomaly', s=16)

    # ax[8].plot(test_df['visits'], color='black', label = 'Normal', linewidth=1.5)
    # ax[8].scatter(a.index ,a['visits'], color='red', label = 'Anomaly', s=16)

    # ax[9].plot(test_df['Anomaly_score'], color='black', label = 'Normal', linewidth=1.5)
    # ax[9].scatter(a.index ,a['Anomaly_score'], color='red', label = 'Anomaly', s=16)
    
    # #ax.plot(pd.Series(prediction_score_DL.flatten()*10), color='blue', label = 'Score', linewidth=0.5)

    # plt.legend()
    # #plt.title("Anamoly Detection Using ECOD model")
    # plt.xlabel('Date')
    # #plt.ylabel('booked_revenue')
    # plt.show()


    #return test_df.to_json(orient="records")
    return prepare_test_json(test_df, op)

