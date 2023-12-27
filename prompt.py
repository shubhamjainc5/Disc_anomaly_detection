
reasoning_sys_prompt = """Given a time series table data of product sales, your job is to analyse and create a correct detailed causal reasoning behind some of the anomalous instances in data indicated by anomaly flag in the data."""

reasoning_prompt = """
Below I am providing you table data in comma seperated format :
{table_data}

here is just an example to show how the casual reasoning for anomalous points should look like:
'''
"AnomalyReasoning":[

"anomaly_date":"2021-12-13",
"reason": "This week shows a higher shipped revenue compared to previous weeks. This could be due to the holiday season where people tend to buy more. The return units are also significantly higher, which could be due to more purchases leading to more returns.",

"anomaly_date":"2021-11-29",
"reason": "This week also shows a high booked revenue, though not as high as the previous week. This could be due to the continued effect of the sales events in the previous week. The shipped units are also significantly higher than the previous week, indicating more products were sold and shipped.",

            ]
'''

Please make sure you always abide by following rules:
- You MUST predict detailed casual reasoning for each anomalous points only based on the provided data only. 
- You SHOULD try to predict reasoning in maximum 100 to 300 words only.
- Remember you MUST only provide causal reasoning by checking the correlation of other column values. 
- ALWAYS think step by step for predicting correct causal reasoning.

\n{format_instructions}
Output JSON:
"""

