
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

optimization_sys_prompt = """Given a time series forcasted table data of product sales, your job is to use your knowledge to first identify the reason for the predictive performance and then suggest corrective measures (actions which an organization can take to improve the respective kpi performance)."""

optimization_prompt = """
Below I am providing you forecasted table data in comma seperated format :
{table_data}

Above table contains future data for a KPI, KPI's 6 week's moving window average and the percentage of difference between kpi's actual value and kpi's moving window average value.

you MUST only provide reasning and corrective measures only for below dates only:
{anomaly_dates}

here is just an example to show how the narrative for anomalous points should look like:
'''
"ForecastNarrative":[
Dict(
"anomaly_date":"2022-12-26",
"narrative": "Based on the historical trend, Shipped Revenue value for the week starting with (2022-12-26) is expected to be arround $41.64M which would be -37.07% below past 6 weeks moving average. This downward trend could be because of seasonal variations,reduced customer spending and increased competition. you can take few actions like Plan marketing campaigns and promotions leading up to holidays, Analyze the competition and adjust pricing or offer unique promotions and refactor customer feedback & preferences on top of previous holiday marketing strategies.",
"narrative_html": "Based on the historical trend,<span style='font-weight:bold;'>Shipped Revenue</span> value for the week starting with (<span style='font-weight:bold;'>2022-12-26</span>) is expected to be arround <span style='font-weight:bold;'>$41.64M</span> which would be <span style='color:red;font-weight:bold;'>37.07</span>% below past 6 weeks moving average. This downward trend can be because of seasonal variations,reduced customer spending and increased competition. you can take few actions like Plan marketing campaigns and promotions leading up to holidays, Analyze the competition and adjust pricing or offer unique promotions and refactor customer feedback & preferences on top of previous holiday marketing strategies."
),
Dict(
"anomaly_date":"2023-04-03",
"narrative": "The predicted shipped revenue for the week starting on 2023-04-03 is $39.86M, which is 16.36% below the average of the past 6 weeks. This decrease in revenue could be due to various factors such as changes in customer demand, market dynamics, or external economic conditions. To mitigate this, the organization can take several corrective measures. They can conduct market research to understand customer needs and preferences better. Optimizing the product portfolio to align with market trends and implementing targeted marketing campaigns can help attract new customers and retain existing ones. It is also important to monitor competitor strategies and adjust pricing or promotions accordingly.",
"narrative_html": "The predicted <span style='font-weight:bold;'>Shipped Revenue</span> for the week starting on <span style='font-weight:bold;'>2023-04-03</span> is <span style='font-weight:bold;'>$39.86M</span>, which is <span style='color:red;font-weight:bold;'>16.36%</span> below the average of the past 6 weeks. This decrease in revenue could be due to various factors such as changes in customer demand, market dynamics, or external economic conditions. To mitigate this, the organization can take several corrective measures. They can conduct market research to understand customer needs and preferences better. Optimizing the product portfolio to align with market trends and implementing targeted marketing campaigns can help attract new customers and retain existing ones. It is also important to monitor competitor strategies and adjust pricing or promotions accordingly."
)
            ]
'''

Please make sure you always abide by following rules:
- You MUST also provide reasoning for the forecasted behaviour why given kpi performance could have got dropped.
- You MUST also provide list of actions(only for anomalous data points) which should clearly convey that what corrective measures can be adopted to avoid this drop in future. 
- You MUST use different variation of words like 'increase/spike/rise' and 'decrease/drop' to indicate increase or decrease behaviour of kpi.
- You MUST not use exact narrative given in above example.


Please make sure you always abide by the following rules to create narrative in html format:
- ALWAYS allowed to use two colors [red, green] only.
- ALWAYS use red color for highlighting decreasing % numeric values only.
- ALWAYS use green color for highlighting increasing % numeric values only.
- ALWAUS use bold tags for highlighting metrics, dimensions and dates values only.
- Be sure you MUST NOT change the actual meaning of narrative while converting it into html format.

\n{format_instructions}
Output JSON:
"""