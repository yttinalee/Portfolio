import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import RegressionResults
from describtion_func import describes
path = 'D:/Code_practice/project/MA_1'
raw_data = (f'{path}/marketing_data.csv')
data = pd.read_csv(raw_data)
a = 'Kidhome'
b = 'Teenhome'
lists = [a, b]
print(lists)
# data_groupby_Kidhome = data.groupby(by=[a])
# data_groupby_Teenhome = data.groupby(by=[b])
# print(data)
# print(data.shape)
# print(type(data))
# print(data.head())
# print(data.describe())
# print(data.count())
# print(data.info)
# print(data.columns)
# print(len(data.columns))
# sns.pairplot(data_plot, kind='reg', plot_kws={'line_kws':{'color':'cyan'}}, corner=True)
# plt.show()
# kid_value_counts = data[a].value_counts()
# # print(kid_value_counts)
# teen_value_counts = data[b].value_counts()
# print(teen_value_counts)
# print(products_list.names)

results = describes(datas=data, lists=lists)

# plt.figure(1)
# plt.bar(x=data["kidhome"]["Country"], y=round(data_groupby[kk].sum(),2))
# # plt.bar(nums_sorted, )
# plt.title("Response of Campain in every Countries")
# # plt.figure(2)
# # ax = sns.boxplot(x=nums_sorted, y=groupby_drop_sumsum, data=cmp, color='#99c2a2')
# # ax = sns.swarmplot(x=nums_sorted, y=groupby_drop_sumsum, data=cmp, color='#7d0013')
# plt.show()


# plt.figure("product _ sum")
# plt.bar(x=products_list, y=groupby_drop_sum)
# plt.title("Total Sales in Countries")