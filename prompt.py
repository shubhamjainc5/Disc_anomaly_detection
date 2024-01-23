
reasoning_sys_prompt = """Given a historical time series table data of product sales, your job is to create a narrative for the given data."""

reasoning_prompt = """
Below I am providing you table data in comma seperated format :
{table_data}

Above table contains data for different KPIs wow(week on week) percentage change( indicating increase/spike/rise or decrease/drop) of respective kpis.

here is just an example to show how the narrative for given data points should look like:
'''
"AnomalyReasoning":[
Dict(
"anomaly_date":"2021-12-13",
"reason": "The week starting from (2021-12-13) saw a spike of 17% in Shipped Revenue compared to previous weeks and the Return units also saw significant increase of 5%.",
"reason_html": "The week starting from (<span style='font-weight:bold;'>2021-12-13</span>) saw a spike of <span style='color:green;font-weight:bold;'>17</span>% in <span style='font-weight:bold;'>Shipped Revenue</span> compared to previous weeks and the <span style='font-weight:bold;'>Return units</span> also saw significant increase of <span style='color:green;font-weight:bold;'>5</span>%."
),
Dict(
"anomaly_date":"2021-11-29",
"reason": "The week starting with (2021-11-29) saw a 11% rise in Booked Revenue compared to 7% rise of previous week. The Shipped Units also saw a significant jump of 9% as compared to previous week.",
"reason_html": "The week starting with (<span style='font-weight:bold;'>2021-11-29</span>) saw a <span style='color:green;font-weight:bold;'>11</span>% rise in <span style='font-weight:bold;'>Booked Revenue</span> compared to <span style='color:green;font-weight:bold;'>7</span>% rise of previous week. The <span style='font-weight:bold;'>Shipped Units</span> also saw a significant jump of <span style='color:green;font-weight:bold;'>9</span>% as compared to previous week."
)
            ]
'''

Please make sure you always abide by following rules:
- You MUST provide narrative for each data point only based on the provided data only like shown in above examples. 
- You MUST not try to provide reasoning for the performance behaviour .
- You MUST use different variation of words like 'increase/spike/rise' and 'decrease/drop' to indicate increase or decrease behaviour of kpi. 

Please make sure you always abide by the following rules to create reasoning in html format:
- ALWAYS allowed to use two colors [red, green] only.
- ALWAYS use red color for highlighting decreasing % numeric values only.
- ALWAYS use green color for highlighting increasing % numeric values only.
- ALWAUS use bold tags for highlighting metrics, dimensions and dates values only.
- Be sure you MUST NOT change the actual meaning of narrative while converting it into html format.

\n{format_instructions}
Output JSON:
"""

forecasting_sys_prompt = """Given a time series forcasted table data of product sales, your job is to create a narrative for the given data."""

forecasting_prompt = """
Below I am providing you forecasted table data in comma seperated format :
{table_data}

Above table contains future data for a KPI, KPI's moving window average  and the percentage of difference between kpi's actual value and kpi's moving window average value.

here is just an example to show how the narrative for anomalous points should look like:
'''
"ForecastNarrative":[
Dict(
"anomaly_date":"2021-12-13",
"narrative": "The Shipped Revenue is expected to (drop) by -14.04% below the average during the week starting (2021-12-13). Its predicted value for that week is $19.2M.",
"narrative_html": "The <span style='font-weight:bold;'>Shipped Revenue</span> is expected to (<span style='font-weight:bold;'>drop</span>) by <span style='color:red;font-weight:bold;'>14.04</span>% below the average during the week starting (<span style='font-weight:bold;'>2021-12-13</span>). Its predicted value for that week is <span style='font-weight:bold;'>$19.2M</span>."
),
Dict(
"anomaly_date":"2021-11-29",
"narrative": "The Shipped Revenue for the week starting on 2021-11-29 is expected to be -15.46% below the average Shipped Revenue. The average Shipped Revenue for the previous week is $9.86M while the predicted Shipped Revenue for this week is $8.33M.",
"narrative_html": "The <span style='font-weight:bold;'>Shipped Revenue</span> for the week starting on <span style='font-weight:bold;'>2021-11-29</span> is expected to be <span style='color:red;font-weight:bold;'>15.46</span>% below the average <span style='font-weight:bold;'>Shipped Revenue</span>. The average <span style='font-weight:bold;'>Shipped Revenue</span> for the previous week is <span style='font-weight:bold;'>$9.86M</span> while the predicted <span style='font-weight:bold;'>Shipped Revenue</span> for this week is <span style='font-weight:bold;'>$8.33M</span>."
)
            ]
'''

Please make sure you always abide by following rules:
- You MUST provide narrative for each data point only based on the provided data only like shown in above examples with some variations. 
- You MUST not try to provide reasoning for the performance behaviour.
- You MUST use different variation of words like 'increase/spike/rise' and 'decrease/drop' to indicate increase or decrease behaviour of kpi.

Please make sure you always abide by the following rules to create narrative in html format:
- ALWAYS allowed to use two colors [red, green] only.
- ALWAYS use red color for highlighting decreasing % numeric values only.
- ALWAYS use green color for highlighting increasing % numeric values only.
- ALWAUS use bold tags for highlighting metrics, dimensions and dates values only.
- Be sure you MUST NOT change the actual meaning of narrative while converting it into html format.

\n{format_instructions}
Output JSON:
"""

