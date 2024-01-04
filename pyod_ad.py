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
import traceback


with open("config.json", "r") as f:
    domain_config = json.load(f)

DB_CREDS = domain_config["db_creds"]

def merge_quarter_entries(df, kpi_cols):
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


def prepare_test_json(test_df, llm_op, kpi_cols):
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
        narr_html = '' if match_idx==-1 else llm_op[match_idx]['reason_html']

        dict_row = {
            "week_start":row["week_start"],
            "kpi_values": value_dict,
            "narrative":narr,
            "narrativeHtml":narr_html,
            "Anomaly_flag":row["Anomaly_flag"],
            "Anomaly_score":row["Anomaly_score"],
        }

        rows_list.append(dict_row)
    
    return rows_list

def add_wow_vars(narrative_df:pd.DataFrame, kpi_cols:list)->pd.DataFrame:
    
    narrative_vars = copy.deepcopy(narrative_df)
    narrative_vars = narrative_vars.sort_index()
    start_index = narrative_vars.index[0]
    
    for idx in narrative_vars.index:

        if idx==start_index:
            prev_idx = idx
            pass
        else:
            for col in kpi_cols:
                prev_value = narrative_vars.loc[prev_idx,col]
                curr_value = narrative_vars.loc[idx,col]
                perc_change = str( round(100*(curr_value-prev_value)/prev_value,1) )+'%'
                narrative_vars.loc[idx,col+'_wow_change'] = perc_change
            prev_idx = idx

    new_order_cols = ['week_start']
    for col in kpi_cols:
        new_order_cols.append(col)
        new_order_cols.append(col+'_wow_change')
    
    narrative_vars = narrative_vars[new_order_cols]
        
    return narrative_vars


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


def inference_model_result(requestId : str, kpi_cols:list, use_cache:bool):


    try:
        sql_query = "SELECT * FROM spt_anomaly_data"
        df, _ = _execute_sql_query(sql_db, sql_query)

        #apply columns filter
        df = df[['week_start']+ kpi_cols]

        merged_df = merge_quarter_entries(df, kpi_cols)

        narrative_df = copy.deepcopy(merged_df)
        narrative_vars = add_wow_vars(narrative_df, kpi_cols)

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
        narrative_vars = narrative_vars.loc[(merged_df['week_start'] >= start_date)]

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

        anomaly_dates = list(test_df[y_pred_test==1]['week_start'])

        if len(anomaly_dates)>0:
            #filt_narrative_vars = copy.deepcopy(narrative_vars)
            remove_kpi_cols = ["week_start"]+ [ col for col in narrative_vars.columns if "wow" in col ] 
            filt_narrative_vars = narrative_vars[y_pred_test==1][remove_kpi_cols]
            op ,api_cnt, api_tokens, llm_status = generate_reasoning(filt_narrative_vars.to_csv(), anomaly_dates, use_cache)
            Logger.info(f"{api_cnt} API Calls were made with an average of {api_tokens} tokens per call for narrative generation")

            if llm_status in ["LLMParsed","LLMRetryParsed"]:
                status_code = 200
                status_msg = "Successfully generated the narrative for the predicted anomalies"
            elif llm_status in ["LLMServerError","UnknownError"]:
                status_code = 300
                status_msg = "LLM Server not responding"
            else:
                status_code = 500
                status_msg = "Anomaly service server failed"

            print(op)
        else:
            op = []
            status_code = 200
            status_msg = "there are no anomalies present in anomaly table"

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

        response = prepare_test_json(test_df, op, kpi_cols)

        response_dict = {"status_code":status_code, "status_msg":status_msg, "data":response, "error":""}
    
    except Exception as e:
        Logger.error(traceback.format_exc())
        response = []
        status_code = 500
        status_msg = "Anomaly service server failed"
        response_dict = {"status_code":status_code, "status_msg":status_msg, "data":response, "error":str(traceback.format_exc())}
    
    return response_dict

