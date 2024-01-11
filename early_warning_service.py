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

from openai_processor_earlywarning import generate_ew_narrative
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

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

def prepare_earlywarning_json(test_df, llm_op, sel_kpi):
    test_df = copy.deepcopy(test_df)
    llm_anomaly_dt = [ op['anomaly_date'] for op in llm_op]
    rows_list = []
    for idx, row in test_df.iterrows():

        match_idx = llm_anomaly_dt.index(row["week_start"]) if row["week_start"] in llm_anomaly_dt else -1
        narr = '' if match_idx==-1 else llm_op[match_idx]['narrative']
        narr_html = '' if match_idx==-1 else llm_op[match_idx]['narrative_html']

        dict_row = {
            "week_start":row["week_start"],
            "kpi": sel_kpi,
            "value": int(row[sel_kpi]),
            "avg_value": int(row['avg_'+sel_kpi]),
            "lower_threshold": int(row['lb_'+sel_kpi]),
            "upper_threshold": int(row['ub_'+sel_kpi]),
            "narrative":narr,
            "narrativeHtml":narr_html,
            "Forecasted":row["Forecasted"],
            "forecast_anomaly":row["forecast_anomaly"],
        }

        rows_list.append(dict_row)
    
    return rows_list

def run_early_warning(requestId : str, sel_kpi:str, use_cache:bool):


    try:
        sql_query = "SELECT * FROM spt_anomaly_data"
        df, _ = _execute_sql_query(sql_db, sql_query)

        df = df[['week_start']+ [sel_kpi]]

        merged_df = merge_quarter_entries(df, [sel_kpi])

        merged_df['week_start'] = merged_df['week_start'].apply(lambda x: pd.to_datetime(x))
        merged_df = merged_df.set_index('week_start')

        current_period = 1
        forecast_period = 1

        model_fit = Holt(merged_df).fit()
        print(model_fit.model.params)
        pred_values = model_fit.forecast(forecast_period*4)
        pred_values.name=sel_kpi
        pred_df = pred_values.reset_index().rename(columns={"index":"week_start"})
        merged_df['Forecasted'] = False
        pred_df['Forecasted'] = True
        final_df = pd.concat([merged_df.reset_index(),pred_df], ignore_index=True)

        window_num = 6
        tol = 0.8
        avg_kpi = final_df[sel_kpi].rolling(window=window_num).mean().dropna()
        dev_kpi = final_df[sel_kpi].rolling(window=window_num).std().dropna()
        final_df['avg_'+sel_kpi] = avg_kpi
        final_df['ub_'+sel_kpi] = avg_kpi + tol*dev_kpi
        final_df['lb_'+sel_kpi] = avg_kpi - tol*dev_kpi
        final_df['dev_'+sel_kpi] = dev_kpi
        final_df = final_df.dropna()
        final_df['diff_avg_perc'] =  round( 100*(final_df[sel_kpi] - final_df['avg_'+sel_kpi] )/final_df['avg_'+sel_kpi] , 2)
        final_df['diff_avg_perc'] = final_df['diff_avg_perc'].apply(lambda x: str(x)+'%')

        final_df['forecast_anomaly'] = (final_df[sel_kpi]<final_df['lb_'+sel_kpi]) | (final_df[sel_kpi]>final_df['ub_'+sel_kpi])
        final_df['forecast_anomaly'] = final_df['forecast_anomaly'] & final_df['Forecasted']

        final_df = final_df[-(current_period*4+forecast_period*4):]

        # plt.plot(final_df['week_start'], final_df[sel_kpi], marker='o')
        # plt.plot(final_df['week_start'], final_df['avg_'+sel_kpi], color='blue', linestyle='--')
        # plt.plot(final_df['week_start'], final_df['ub_'+sel_kpi], color='red', linestyle='--')
        # plt.plot(final_df['week_start'], final_df['lb_'+sel_kpi], color='green', linestyle='--')
        # plt.show()
        # plt.savefig('forecast.png')

        final_df['week_start'] = final_df['week_start'].dt.strftime('%Y-%m-%d')

        anomaly_dates = list(final_df[final_df['forecast_anomaly']==True]['week_start'])

        if len(anomaly_dates)>0:

            filt_narrative_vars = final_df[final_df['forecast_anomaly']==True][['week_start',sel_kpi,'avg_'+sel_kpi,'diff_avg_perc']]
            op ,api_cnt, api_tokens, llm_status = generate_ew_narrative(filt_narrative_vars.to_csv(), anomaly_dates, use_cache)
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
            #plot_charts(test_df, kpi_cols, y_pred_test)
        else:
            op = []
            status_code = 200
            status_msg = "there are no anomalies present in anomaly table"

        response = prepare_earlywarning_json(final_df, op, sel_kpi)
        status_code = 200
        status_msg = "Successfully generated the narrative for the forecasted value"

        response_dict = {"status_code":status_code, "status_msg":status_msg, "data":response, "error":""}
    
    except Exception as e:
        Logger.error(traceback.format_exc())
        response = []
        status_code = 500
        status_msg = "Early warning service server failed"
        response_dict = {"status_code":status_code, "status_msg":status_msg, "data":response, "error":str(traceback.format_exc())}
    
    return response_dict

