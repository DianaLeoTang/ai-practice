'''
Author: Diana Tang
Date: 2025-02-23 22:02:17
LastEditors: Diana Tang
Description: some description
FilePath: /ai-practice/basic_part/ldfx.py
'''
# pip install plotly

# import plotly.express as px
# fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
# fig.show()

import plotly.express as px
data = dict(
    number=[39, 27.4, 20.6, 11, 2],
    stage=["Website visit", "Downloads", "Potential customers", "Requested price", "invoice sent"])
fig = px.funnel(data, x='number', y='stage')
fig.show()
# data = dict( #准备漏斗数据
#     number=[59, 32, 18, 9, 2],
#     stage=["访问数", "下载数", "注册数", "搜索数", "付款数"])
# fig = px.funnel(data, x='number', y='stage') #把数据传进漏斗图
# fig.show() #显示漏斗图