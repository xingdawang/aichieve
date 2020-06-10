#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)

source = pd.read_excel('南京情况.xlsx')
print('原始数据预览:')
print('数据尺度:', source.shape)
print(source.head())
print()



print('数据探索')
print('所有参数：', list(source.columns))
noisy_features = ['标题', '关注', '总价', '平方数', '小区', '商品类型', '房屋用途', '梯户', '地铁线',
                  '幼儿园', '小学', '中学', '商场', '市场', '菜市场', '公园', '电影院', '标题链接']
print('无用参数：', noisy_features)
print()



# 舍弃无用信息列
source = source.drop(columns=noisy_features)
print('舍弃无用信息列后，数据预览')
print('数据尺度:', source.shape)
print(source.head())
print()


# ### 线性回归

# ### Feature Engineering 数据工程


from sklearn.preprocessing import LabelEncoder

# 标签列
y = source['单价']

# 处理户型 (OneHotEncoding)
huxing = pd.get_dummies(source['户型'], prefix='户型')

# 处理朝向
all_direction_list = list(set(source['朝向'])) # 提取朝向
all_directions = set(' '.join(all_direction_list).split(' ')) # 将不同朝向用空格分开，用集合去重后再用空格组合
all_directions = list(map(lambda x: '朝向_' + x, list(all_directions))) # 用匿名函数添加‘朝向_’前缀
chaoxiang = pd.DataFrame(columns=all_directions)
# 将卷数据以此填充到新的朝向Data Frame中
direction_index = 0
for row in source['朝向']:
    direction_list = row.split(' ')
    for direction in direction_list:
        chaoxiang.at[direction_index, '朝向_' + direction] = 1
    direction_index += 1
chaoxiang.fillna(0, inplace = True)        

# 处理楼层
floor = pd.DataFrame(columns=['楼层_位置', '楼层_高度'])
floor_index = 0
for row in source['楼层']:
    position, hight_number = row.split('/')
    # 楼层转换成相应数字
    if position == '高楼层':
        position = 3
    elif position == '中楼层':
        position = 2
    elif position == '低楼层':
        position = 1
    floor.at[floor_index, '楼层_位置'] = position
    
    # 提取楼层高度
    hight_number = hight_number.replace('共', '')
    hight_number = hight_number.replace('层', '')
    hight_number = int(hight_number)
    floor.at[floor_index, '楼层_高度'] = hight_number
    floor_index += 1

# 处理装修 (OneHotEncoding)
decoration = pd.get_dummies(source['装修'], prefix='装修')

# 处理楼房类型
house_type = pd.DataFrame(columns=['楼房类型_年份', '楼房类型_板楼', '楼房类型_塔楼', '楼房类型_平房'])
house_type_index = 0
for row in source['楼房类型']:
    
    if row == '暂无数据':
        house_type.at[house_type_index, '楼房类型_平房'] = 0
    elif row == '平房':
        house_type.at[house_type_index, '楼房类型_平房'] = 1
    elif row == '塔楼':
        house_type.at[house_type_index, '楼房类型_塔楼'] = 1
    elif row == '板楼':
        house_type.at[house_type_index, '楼房类型_板楼'] = 1
    elif row == '板塔结合':
        house_type.at[house_type_index, '楼房类型_板楼'] = 1
        house_type.at[house_type_index, '楼房类型_塔楼'] = 1
    elif '年建'in row and '/' not in row:
        house_year = row.replace('年建', '')
        house_type.at[house_type_index, '楼房类型_年份'] = house_year = int(house_year)
    else:
        house_year, house_type_raw = row.split('/')
        house_year = house_year.replace('年建', '')
        house_type.at[house_type_index, '楼房类型_年份'] = house_year = int(house_year)
        if house_type_raw == '塔楼':
            house_type.at[house_type_index, '楼房类型_塔楼'] = 1
        elif house_type_raw == '板楼':
            house_type.at[house_type_index, '楼房类型_板楼'] = 1
        elif house_type_raw == '板塔结合':
            house_type.at[house_type_index, '楼房类型_板楼'] = 1
            house_type.at[house_type_index, '楼房类型_塔楼'] = 1
    house_type_index += 1
house_type = house_type.fillna(0)

# 处理地区 (OneHotEncoding)
area = pd.get_dummies(source['地区'], prefix='地区')

# 处理挂牌时间
onboard_time = source['挂牌时间'].apply(lambda x: int(x[4:8]))

# 处理上次交易时间
last_purchase_time = source['上次交易时间'].apply(lambda x: 0 if x == '上次交易暂无数据' else int(x[4:8]))

# 处理满年限 (LabelEncoding)
full_year = LabelEncoder().fit_transform(source['满年限'])
full_year = pd.Series(full_year, name='满年限')

# 处理抵押情况
mortgage = source['抵押情况'].apply(lambda x: 0 if x == '无抵押' else 1)

# 处理权限
ownership = source['权限'].apply(lambda x: 70 if x == '产权年限70年' else 0)

# 处理小学距离
missing_max_distance = 3000
primary_distance = source['小学距离'].fillna(str(missing_max_distance)+'米')
primary_distance = primary_distance.apply(lambda x: int(x.replace('米', '')))





# 组合数据保存结果
data = pd.concat([huxing, chaoxiang], axis=1)
data = pd.concat([data, floor], axis=1)
data = pd.concat([data, decoration], axis=1)
data = pd.concat([data, house_type], axis=1)
data = pd.concat([data, area], axis=1)
data = pd.concat([data, onboard_time], axis=1)
data = pd.concat([data, last_purchase_time], axis=1)
data = pd.concat([data, full_year], axis=1)
data = pd.concat([data, mortgage], axis=1)
data = pd.concat([data, ownership], axis=1)
data = pd.concat([data, source[['地铁站距离', '幼儿园距离']]], axis=1)
data = pd.concat([data, primary_distance], axis=1)
data = pd.concat([data, source[['中学距离', '商场距离', '市场距离', '菜市场距离', '公园距离', '电影院距离']]], axis=1)
data.fillna(missing_max_distance, inplace = True)
print('线性回归数据处理后预览')
print('数据尺度:', data.shape)
print(data.head())
print()



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(data)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

component_number = 40
print('主成分分析(PCA), 主成分数:', component_number)
component_pct = sum(pca.explained_variance_ratio_[:component_number]) * 100
print('PCA保留率: %.2f%%' % component_pct)
print()




pca = PCA(n_components=40)
X_pca = pca.fit_transform(X_scaled)
print('PCA数据变换后尺度:', X_pca.shape)
print()



from sklearn.model_selection import train_test_split

# 分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=60)
print('分训练集和测试集分类')
print()



from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
en = ElasticNetCV(cv = 10)
print('训练ElasticNet线性模型')
params = en.fit(X_train, y_train)
print('ElasticNet线性模型参数:', params)
print()




r2 = en.score(X_test, y_test)
print('ElasticNet R^2:', r2)
y_pred = en.predict(X_test)
rmse = mean_squared_error(y_pred, y_test)
print('ElasticNet RMSE:', rmse)
print()




from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


print('训练线性回归(Linear Regression)模型')
r2 = lin_reg.score(X_test, y_test)
print('Linear Regression R^2:', r2)
y_pred = y_pred = lin_reg.predict(X_test)
rmse = mean_squared_error(y_pred, y_test)
print('Linear Regression RMSE:', rmse)
print()


# ### 决策树模型

# ### Feature Engineering 数据工程




# 标签列
y = source['单价']

# 处理楼房类型
house_type = pd.DataFrame(columns=['楼房类型_年份', '楼房类型_类型'])
house_type_index = 0
for row in source['楼房类型']:
    if row in ['塔楼', '平房', '暂无数据', '板塔结合', '板楼']:
        house_type.at[house_type_index, '楼房类型_年份'] = '暂无数据'
        house_type.at[house_type_index, '楼房类型_类型'] = row
    elif '年建' in row and '/' not in row:
        house_type.at[house_type_index, '楼房类型_年份'] = row
        house_type.at[house_type_index, '楼房类型_类型'] = '暂无数据'
    else:
        year, house_type_raw = row.split('/')
        house_type.at[house_type_index, '楼房类型_年份'] = year
        house_type.at[house_type_index, '楼房类型_类型'] = house_type_raw
    house_type_index += 1




import copy
data = pd.concat([source['户型'], source['朝向']], axis = 1)
data = pd.concat([data, floor], axis = 1)
data = pd.concat([data, house_type], axis = 1)
data = pd.concat([data, source['地区']], axis = 1)
data = pd.concat([data, onboard_time], axis = 1)
data = pd.concat([data, last_purchase_time], axis = 1)
data = pd.concat([data, mortgage], axis = 1)
data = pd.concat([data, ownership], axis = 1)
data = pd.concat([data, source[['地铁站距离', '幼儿园距离']]], axis=1)
data = pd.concat([data, primary_distance], axis=1)
data = pd.concat([data, source[['中学距离', '商场距离', '市场距离', '菜市场距离', '公园距离', '电影院距离']]], axis=1)
data.fillna(missing_max_distance, inplace = True)

# 手工输入存档
data_copied = copy.deepcopy(data)


category_column = ['户型', '朝向', '楼层_位置', '楼层_高度', '楼房类型_年份', '楼房类型_类型', '地区', '挂牌时间',
                  '上次交易时间', '抵押情况', '权限']
category_dict = {}
for category in category_column:
    le = LabelEncoder()
    category_dict[category] = le.fit(data[category])
    data[category] = le.transform(data[category])

print('树型、集合型处理后')
print('数据尺度', data.shape)
print('数据预览', data.head())
print()
# data




# 分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=60)





from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(max_depth=5)
dt_reg.fit(X_train, y_train)
r2 = dt_reg.score(X_test, y_test)
print('训练决策树模型(Decision Tree)')
print('Decision Tree R^2:', r2)
y_pred = dt_reg.predict(X_test)
rmse = mean_squared_error(y_pred, y_test)
print('Decision Tree RMSE:', rmse)
print()




print('特征名称：', list(data.columns))
print('特征百分比重要性：', dt_reg.feature_importances_)
temp = list(dt_reg.feature_importances_)
featue_important_index_list = [temp.index(x) for x in temp if x != 0]
print('非0特征百分比重要性：')

result = {}
for index in featue_important_index_list:
    result[data.columns[index]] = temp[index]*100

for key in sorted(result, key=result.get, reverse = True):
    print('%s %.2f%%' % (key, result[key]))
print()




# 随机森林
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=500)
rf_reg.fit(X_train, y_train)
r2 = rf_reg.score(X_test, y_test)
print('训练随机森林模型(Random Forest)')
print('Random Forest R^2:', r2)
y_pred = rf_reg.predict(X_test)
rmse = mean_squared_error(y_pred, y_test)
print('Random Forest RMSE:', rmse)
print()


# ### 手动输入房屋商品属性



manual_input = {}

# 手动输入房屋商品属性
print('手动输入房屋商品属性')
for column in data_copied.columns:
    if column in category_column:
        print(column + ' - 可选:', set(data_copied[column]))

    temp = input(column + ':')
    if temp.isdigit():
        manual_input[column] = [float(temp)]
    else:
        manual_input[column] = [temp]
    




manual_df = pd.DataFrame.from_dict(manual_input)
# label encoding
for column in category_dict:
    manual_df[column] = category_dict[column].transform(manual_df[column])




# 预测房价
predicted_price = rf_reg.predict(manual_df)
print('预测房价：', predicted_price[0])





