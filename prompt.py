
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

# html_sys_prompt = """Given a casual insight data, your job is to analyse and create a correct detailed causal reasoning behind some of the anomalous instances in data indicated by anomaly flag in the data."""

# html_beautify_prompt = """Given an insight, beautify it with HTML tags based on the provided rules.

# Please make sure you always abide by the following rules:
# - ALWAYS allowed to use two colors [red, green] only.
# - ALWAYS use red color for highlighting decreasing % numeric values only.
# - ALWAYS use green color for highlighting increasing % numeric values only.
# - ALWAUS use bold tags for highlighting metrics, dimensions and dates values only.
# - ALWAYS use ordered list tags for numeric pointers as (1., 2., 3.) whenever necessary.

# Be sure you MUST NOT change the actual meaning of insight.

# Insight:
# ```
# The data shows the trend of Shipped Revenue in the United States from 2023-07-03 to 2023-08-21. The Shipped Revenue for the Last week (starting 2023-08-21) was $23.66 M, which is a decrease of 4.4% compared to the immediate previous week (starting 2023-08-14). Moreover,

# 1. The Shipped Revenue for the Last week shows an increase of 59.5% compared to the first week (starting 2023-07-03) revenue $14.83 M.
# 2. The Shipped Revenue for the Last week shows an increase of 15.8% compared to the second week (starting 2023-07-10) revenue $20.43 M.
# 3. The Shipped Revenue for the Last week shows an increase of 5.8% compared to the third week (starting 2023-07-17) revenue $22.36 M.
# 4. The Shipped Revenue for the Last week shows a decrease of 6.1% compared to the fourth week (starting 2023-07-24) revenue $25.19 M.
# 5. The Shipped Revenue for the Last week shows an increase of 14.9% compared to the fifth week (starting 2023-07-31) revenue $20.59 M.
# 6. The Shipped Revenue for the Last week shows a decrease of 2.9% compared to the sixth week (starting 2023-08-07) revenue $24.37 M.

# Overall, the average Shipped Revenue for all weeks is $22.02 M and a standard deviation of $3.20 M. The trend of Shipped Revenue in the United States for the last week is 4.4% lower than the immediate previous week, indicating a downward trend.
# ```
# AI: <p>The data shows the trend of <b>Shipped Revenue</b> in the <b>United States</b> from <b>2023-07-03</b> to <b>2023-08-21</b>. The <b>Shipped Revenue</b> for the last week (starting <b>2023-08-21</b>) was <span style='color:red;'>$23.66 M</span>, which is a decrease of <span style='color:red;'>4.4%</span> compared to the immediate previous week (starting <b>2023-08-14</b>). Moreover,</p> <br> <ol> <li>The <b>Shipped Revenue</b> for the last week shows an increase of <span style='color:green;'>59.5%</span> compared to the first week (starting <b>2023-07-03</b>) revenue $14.83 M. </li> <br> <li>The <b>Shipped Revenue</b> for the last week shows an increase of <span style='color:green;'>15.8%</span> compared to the second week (starting <b>2023-07-10</b>) revenue $20.43 M. </li> <br> <li>The <b>Shipped Revenue</b> for the last week shows an increase of <span style='color:green;'>5.8%</span> compared to the third week (starting <b>2023-07-17</b>) revenue $22.36 M. </li> <br> <li>The <b>Shipped Revenue</b> for the last week shows a decrease of <span style='color:red;'>6.1%</span> compared to the fourth week (starting <b>2023-07-24</b>) revenue $25.19 M. </li> <br> <li>The <b>Shipped Revenue</b> for the last week shows a increase of <span style='color:green;'>14.9%</span> compared to the fifth week (starting <b>2023-07-31</b>) revenue $20.59 M. </li> <br> <li>The <b>Shipped Revenue</b> for the last week shows a decrease of <span style='color:red;'>2.9%</span> compared to the sixth week (starting <b>2023-08-07</b>) revenue $24.37 M. </li> </ol><br><p>Overall, the average <b>Shipped Revenue</b> for all weeks is <span style='color:green;'>$22.02 M</span> and a standard deviation of <span style='color:green;'>$3.20 M</span>. The trend of <b>Shipped Revenue</b> in the <b>United States</b> for the last week is <span style='color:red;'>4.4%</span> lower than the previous dates, indicating a downward trend.</p>

# Insight:
# ```
# The data shows the trend of Shipped Revenue in the United States from 2023-07-03 to 2023-08-21. The Shipped Revenue for the last week (2023-08-21) was $23.65 M, a decrease of 4.4% compared to the immediate previous week (starting 2023-08-14). The average Shipped Revenue for all weeks is $22.84 M with a standard deviation of $7.84 M. The trend of Shipped Revenue for the last week is lower than the immediate previous week, indicating a negative trend.
# ```
# AI: <p>The data shows the trend of <b>Shipped Revenue</b> in the <b>United States</b> from <b>2023-07-03</b> to <b>2023-08-21</b>. The <b>Shipped Revenue</b> for the last week (<b>2023-08-21</b>) was <span style='color:red;'>$23.65 M</span>, a decrease of <span style='color:red;'>4.4%</span> compared to the immediate previous week (starting <b>2023-08-14</b>). The average <b>Shipped Revenue</b> for all weeks is <span style='color:green;'>$22.84 M</span> with a standard deviation of <span style='color:green;'>$7.84 M</span>. The trend of <b>Shipped Revenue</b> for the last week is lower than the immediate previous week, indicating a negative trend.</p>

# Insight:
# ```
# The Shipped Revenue in the United States for the Last Year (2023-01-01) has increased compared to the previous year (2022-01-01 to 2022-12-31). This can be attributed to the increase in Booked Revenue ($727.64 M) for the Last year, decrease in Cancel Revenue ($36.47 M) for the Last year and increase in Visits (43.32 M) for the Last year. 
# Overall, these factors have caused the Shipped Revenue to increase for the Last Year compared to the previous year.
# ```
# AI: <p>The <b>Shipped Revenue</b> in the <b>United States</b> for the Last Year (<b>2023-01-01</b>) has increased compared to the previous year (<b>2022-01-01</b> to <b>2022-12-31</b>). This can be attributed to the increase in <b>Booked Revenue</b> (<span style='color:green;'>$727.64 M</span>) for the Last year, decrease in <b>Cancel Revenue</b> (<span style='color:red;'>$36.47 M</span>) for the Last year and increase in <b>Visits</b> (<span style='color:green;'>43.32 M</span>) for the Last year.</p><br><p>Overall, these factors have caused the <b>Shipped Revenue</b> to increase for the Last Year compared to the previous year.</p>

# Insight:
# ```
# {insight}
# ```
# AI:
# """

