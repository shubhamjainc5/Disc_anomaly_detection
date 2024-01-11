
reasoning_sys_prompt = """Given a time series table data of product sales, your job is to analyse and create a correct detailed causal reasoning behind some of the anomalous instances in data indicated by anomaly flag in the data."""

reasoning_prompt = """
Below I am providing you table data in comma seperated format :
{table_data}

Above table contains data for different KPIs wow(week on week) percentage change( indicating increase/spike/rise or decrease/drop) of respective kpis.

the anomalous points detected in above table data occurs at below dates, your analysis and reasoning should strictly be done at below dates only.
{anomaly_dates}

here is just an example to show how the casual reasoning for anomalous points should look like:
'''
"AnomalyReasoning":[
Dict(
"anomaly_date":"2021-12-13",
"reason": "The week starting from (2021-12-13) saw a spike of 17% in shipped revenue compared to previous weeks. This could be due to the holiday season where people tend to buy more. The return units also saw significant increase of 5%, which could be due to more purchases leading to more returns.",
"reason_html": "The week starting from (<span style='font-weight:bold;'>2021-12-13</span>) saw a spike of <span style='color:green;font-weight:bold;'>17</span>% in <span style='font-weight:bold;'>Shipped Revenue</span> compared to previous weeks. This could be due to the holiday season where people tend to buy more. The <span style='font-weight:bold;'>Return units</span> also saw significant increase of <span style='color:green;font-weight:bold;'>5</span>%, which could be due to more purchases leading to more returns."
),
Dict(
"anomaly_date":"2021-11-29",
"reason": "The week starting with (2021-11-29) saw a 11% rise in booked revenue compared to 7% rise of previous week. This could be due to the continued effect of the sales events in the previous week. The shipped units also saw a significant jump of 9% as compared to previous week, indicating more products were sold and shipped.",
"reason_html": "The week starting with (<span style='font-weight:bold;'>2021-11-29</span>) saw a <span style='color:green;font-weight:bold;'>11</span>% rise in <span style='font-weight:bold;'>Booked Revenue</span> compared to <span style='color:green;font-weight:bold;'>7</span>% rise of previous week. This could be due to the continued effect of the sales events in the previous week. The <span style='font-weight:bold;'>Shipped Units</span> also saw a significant jump of <span style='color:green;font-weight:bold;'>9</span>% as compared to previous week, indicating more products were sold and shipped."
)
            ]
'''

Please make sure you always abide by following rules:
- You MUST predict detailed casual reasoning for each anomalous points only based on the provided data only. 
- You SHOULD try to predict reasoning in maximum 100 to 300 words only.
- Remember you MUST only provide causal reasoning by checking the correlation of other column values. 
- ALWAYS think step by step for predicting correct causal reasoning.

Please make sure you always abide by the following rules to create reasoning in html format:
- ALWAYS allowed to use two colors [red, green] only.
- ALWAYS use red color for highlighting decreasing % numeric values only.
- ALWAYS use green color for highlighting increasing % numeric values only.
- ALWAUS use bold tags for highlighting metrics, dimensions and dates values only.
- Be sure you MUST NOT change the actual meaning of reasoning while converting it into html format.

\n{format_instructions}
Output JSON:
"""

forecasting_sys_prompt = """Given a time series table data of product sales, your job is to create a narrative for the given data."""

forecasting_prompt = """
Below I am providing you table data in comma seperated format :
{table_data}

Above table contains data for a KPI, KPI's moving window average  and the percentage of difference between kpi's actual value and kpi's moving window average value.

here is just an example to show how the narrative for anomalous points should look like:
'''
"ForecastNarrative":[
Dict(
"anomaly_date":"2021-12-13",
"narrative": "The shipped revenue is expected to (drop) by -14.04% below the average during the week starting (2021-12-13). Its predicted value for that week is 91959.359. Please take required remedial actions.",
"narrative_html": "The <span style='font-weight:bold;'>Shipped Revenue</span> is expected to (<span style='font-weight:bold;'>drop</span>) by <span style='color:red;font-weight:bold;'>14.04</span>% below the average during the week starting (<span style='font-weight:bold;'>2021-12-13</span>). Its predicted value for that week is <span style='font-weight:bold;'>91959.359</span> . Please take required remedial actions."
)
            ]
'''

Please make sure you always abide by following rules:
- You MUST predict detailed narrative for each anomalous points only based on the provided data only. 
- You SHOULD try to predict narrative in maximum 100 to 300 words only.
- ALWAYS think step by step for predicting correct narrative.

Please make sure you always abide by the following rules to create reasoning in html format:
- ALWAYS allowed to use two colors [red, green] only.
- ALWAYS use red color for highlighting decreasing % numeric values only.
- ALWAYS use green color for highlighting increasing % numeric values only.
- ALWAUS use bold tags for highlighting metrics, dimensions and dates values only.
- Be sure you MUST NOT change the actual meaning of narrative while converting it into html format.

\n{format_instructions}
Output JSON:
"""

