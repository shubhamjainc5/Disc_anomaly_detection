from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging
Logger = logging.getLogger('EarlyWarning')
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from typing import Dict, List, Any
from openai_processor_earlywarning import generate_ew_narrative
from numerize import numerize
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX 

from joblib import dump, load
import json
import copy
import time

from database_connector import SQLDataBase
from dateutil.relativedelta import relativedelta
import traceback

with open("config.json", "r") as f:
    domain_config = json.load(f)


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
        Logger.info("SQL to pandas dataframe successfull")
        return df, True
    except Exception as e:
        Logger.error(traceback.format_exc())
        Logger.error("SQL to pandas dataframe failed")
        return None, False


def prepare_earlywarning_json(test_df, llm_op, sel_kpi):
    test_df = copy.deepcopy(test_df)
    llm_anomaly_dt = [ op['anomaly_date'] for op in llm_op]
    rows_list = []
    for idx, row in test_df.iterrows():

        match_idx = llm_anomaly_dt.index(row["week_start"]) if row["week_start"] in llm_anomaly_dt else -1
        narr = '' if match_idx==-1 else llm_op[match_idx]['narrative']
        narr_html = '' if match_idx==-1 else llm_op[match_idx]['narrative_html']

        dict_row = {
            "refreshDate":row["week_start"],
            "kpi": sel_kpi,
            "value": int(row[sel_kpi]),
            "smaBound": int(row['avg_'+sel_kpi]),
            "lowerBound": int(row['lb_'+sel_kpi]),
            "upperBound": int(row['ub_'+sel_kpi]),
            "narrative":narr,
            "narrativeHtml":narr_html,
            "isForecasted":row["Forecasted"],
            "isEarlyWarning":row["forecast_anomaly"],
        }

        rows_list.append(dict_row)
    
    return rows_list

def run_early_warning(requestId : str, sel_kpi:str, use_cache:bool, sql_db:Any):

    status_code = 0
    status_msg= 'forecasting model failed'
    try:
        
        sql_query = "SELECT * FROM spt_anomaly_data"
        df, fetch_flag = _execute_sql_query(sql_db, sql_query)

        if fetch_flag==False:
            status_code = 100
            status_msg = "Database connection failed."
            raise AssertionError(f"Database connection failed.")
    
        if df.shape[0]==0 or df.shape[1]==0:
            Logger.info("the number of observation in sql dataframe is".format(df.shape[0]))
            Logger.info("the number of columns in sql dataframe are".format(str(df.columns)))
            status_code = 100
            status_msg = "Empty historical data in database table"
            raise AssertionError(f"Empty historical data in database table")

        df = df[['week_start']+ [sel_kpi]]

        merged_df = merge_quarter_entries(df, [sel_kpi])

        merged_df['week_start'] = merged_df['week_start'].apply(lambda x: pd.to_datetime(x))
        merged_df = merged_df.set_index('week_start')

        current_period = 1
        forecast_period = 1

        p = domain_config["ew_model_config"][sel_kpi]["p"]
        d = domain_config["ew_model_config"][sel_kpi]["d"]
        q = domain_config["ew_model_config"][sel_kpi]["q"]
        P = domain_config["ew_model_config"][sel_kpi]["P"]
        D = domain_config["ew_model_config"][sel_kpi]["D"]
        Q = domain_config["ew_model_config"][sel_kpi]["Q"]
        m = domain_config["ew_model_config"][sel_kpi]["m"]

        start_time = time.time()
        results = SARIMAX(merged_df[sel_kpi],order=(p, d, q),seasonal_order=(P,D,Q,m)).fit()
        # print(results.summary())
        pred_values = results.forecast(forecast_period*4)
        end_time = time.time()
        Logger.info("model forecasting in early warning service took {:.4f} seconds".format(end_time - start_time))
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
            filt_narrative_vars[sel_kpi] = filt_narrative_vars[sel_kpi].apply(lambda x : '$'+numerize.numerize(x,2))
            filt_narrative_vars['avg_'+sel_kpi] = filt_narrative_vars['avg_'+sel_kpi].apply(lambda x : '$'+numerize.numerize(x,2))
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
        # Logger.exception("Error",e)
        response = []
        status_code = 500 if status_code==0 else status_code
        status_msg = "Anomaly service server failed" if status_code==0 else status_msg
        response_dict = {"status_code":status_code, "status_msg":status_msg, "data":response, "error":str(traceback.format_exc())}
    
    return response_dict

