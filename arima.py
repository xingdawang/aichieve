#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

# read data source
source = pd.read_csv('daoqie_bingtuan_xq.txt', delimiter = ';', header = None)
source.columns = ['法院', '案件', '类型', '评审', '裁判日期', '成员', '法律', '发布日期', '法官', '结果', '其它']

# data preprocessing
source['裁判日期'] = pd.to_datetime(source['裁判日期'])

source


# In[15]:


# Extract year month
source['裁判年月'] = None
for index, row in source.iterrows():
    if row['裁判日期'].month < 10:
        source.loc[index, '裁判年月'] = str(row['裁判日期'].year) + '-0' + str(row['裁判日期'].month)
    else:
        source.loc[index, '裁判年月'] = str(row['裁判日期'].year) + '-' + str(row['裁判日期'].month)

# source['裁判年月'] = pd.to_datetime(source['裁判年月'] )
source


# In[16]:


import matplotlib.pyplot as plt

import matplotlib

# 调整中文字体
matplotlib.rcParams['font.family'] = ['Heiti TC']

# 按照年月聚集案发个数
result = source.groupby('裁判年月').agg({'裁判日期': 'count'})
result.reset_index(inplace = True)

# 分离数据
judge_year_month = result['裁判年月']
judge_case_number = result['裁判日期']

# 绘图
plt.figure(figsize=(15, 12))
plt.plot(judge_year_month, judge_case_number, label='实际数据')
plt.xticks(rotation=-90)
plt.xlabel('裁判年月')
plt.ylabel('案发个数')
plt.legend()
plt.show()


# ## Prediction

# In[17]:


# save actual data source to actual.xlsx
data = result
data.rename(columns = {'裁判日期': '案发个数'}, inplace = True)
data.set_index('裁判年月', inplace = True)
data.to_excel('actual.xlsx') # save result
data


# In[18]:


import statsmodels.api as sm
import itertools
import warnings

warnings.filterwarnings('ignore')

# setARIMA parameters p, d, q in range in [0,2). e.g[0, 1]
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# go through all combination
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data, 
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            print('ARIMA{} x {} - AIC:{}'.format(param, param_seasonal, results.aic)) # Akaike Information Criterion
        except:
            continue


# In[19]:


import statsmodels.api as sm

# bring data and prious determined parameters into the model
mod = sm.tsa.statespace.SARIMAX(data,
                                order=(1, 1, 1),  # seasonality, trend, and noise:
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[20]:


# print diagnos result
results.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[21]:


end_year = 2020
end_month = 0
future_month = 36
predict_year_month = []

# calculate future year-month (index)
for i in range(future_month):
    month = (end_month + i) % 12 + 1
    year = end_year + (end_month + i) // 12
    if month < 10:
        predict_year_month.append(str(year) + '-0' + str(month))
    else:
        predict_year_month.append(str(year) + '-' + str(month))
predict_year_month


# In[22]:


# generate predict table
pred_uc = results.get_forecast(steps=future_month)
pred_ci = pred_uc.conf_int()

pred_df = pred_uc.summary_frame()
pred_df.index = predict_year_month
pred_df.to_excel('prediction.xlsx')
pred_df


# In[23]:


# plot original data and predicted data
plt.figure(figsize=(18, 12))
plt.plot(data, label='实际数据')
plt.plot(pred_df.index, pred_df['mean'], label='预测数据')
plt.fill_between(pred_df.index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1],
                 color='k',
                 alpha=.25)
plt.xlabel('裁判年月')
plt.ylabel('案发个数')
plt.xticks(data.index.tolist() + predict_year_month)
plt.xticks(rotation=-90)
plt.legend()
plt.show()


# In[ ]:




