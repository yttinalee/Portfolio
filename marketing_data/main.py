import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import RegressionResults
import scipy.stats
from collections import OrderedDict
path = 'D:/Code_practice/project/MA_1'
raw_data = (f'{path}/marketing_data.csv')
data = pd.read_csv(raw_data)
data_groupby_country = data.groupby(by=['Country'])
non_products_list = ['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
       'Teenhome', 'Dt_Customer', 'Recency', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Response', 'Complain', 'Country']
products_list = ['MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds']
channels_list = ['NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Response', 'Complain']
basic_info_list = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
       'Teenhome', 'Dt_Customer', 'Recency']
# print(data_groupby.info)
# print(data_groupby['SP'])
# data_plot = data.drop(['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Kidhome',
#        'Teenhome', 'Dt_Customer', 'Recency', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 
#        'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response', 'Complain', 'Country'], axis=1)
# print(data)
# print(data.shape)
# print(type(data))
# print(data.head())
# print(data.describe())
# print(data.count())
# print(data.info)
data_columns = data.columns
print(data.columns)
# print(len(data.columns))
# sns.pairplot(data_plot, kind='reg', plot_kws={'line_kws':{'color':'cyan'}}, corner=True)
# plt.show()

# # # use
nums = set(data["Country"])
nums_sorted = sorted(list(nums))
# print(data["Country"].unique())
# # print(len(data["Country"].unique()))
# length = len(nums)
# print(nums)
# print(sorted(nums))
country_value_counts = data["Country"].value_counts()
# print(max(country_value_counts))
# # # # # use # # #
# # best selling products # # # # # #
# products = data.drop(non_products_list, axis=1)
# print(products.describe())
country_array = np.empty((len(nums),max(country_value_counts)), dtype = str)
# country_array = list()
# print(country_array.shape)
# print(type(country_array[0][0]))
# print(type(data["Country"][0]))
# print(data["Country"][0])
# d = data.index.astype('float64', copy=True)
# print(type(d))
# print(len(d))
# for kk in range(0, len(nums_sorted)):
#   # print(nums_sorted[kk])
#   z = 0
#   for qq in range(0, len(data["Country"])):
#     if nums_sorted[kk] == data["Country"][qq]:
#       country_array[kk].append(data["Country"][z])
#       z = z + 1
# print(country_array[0,0]) 


## Cleaning data -- check for missing values
## find missing
missing = pd.isnull(data).any()
# print(missing)  ## only Income
# # # # Correlations  ## default is Person
# cor1_result = data["PRICE"].corr(data["RM"])
# b = 'MntGoldProds'
# a = 'Income'
# c = 'MntMeatProducts'
# x = data[b]
# y = data[a]
# z = data[c]
# print(type(x),type(y))
# cor1_result = x.corr(y)
# print(cor1_result)
# plt.scatter(x,y)
# plt.xlabel(a)
# plt.ylabel(b)
# # plt.xlim([1930, 2000])
# plt.ylim([0, 200000])
# plt.show()
# regr = LinearRegression.fit(x, y)
# print(regr)
# length = len(data.columns)
# print(length)
# c = 0
# for ii in data_plot.columns:
#   a = 'Income'
#   b = ii
#   x = data[b]
#   y = data[a]  
#   cor1_result = x.corr(y)
#   print(f"{b}: ",cor1_result)
#   plt.figure(c)
#   plt.scatter(x,y)
#   # plt.plot(x, y)
#   c = c + 1

# for features in products_list:
#        # unique_majors = data[features].unique()
#        k2, p = scipy.stats.normaltest(data[features])
#        print(features)
#        print(f"k2: {round(k2,5)}, p: {round(p, 5)}")

# print(unique_majors)
# # use
corr_list = ['Year_Birth', 'Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
total_corr = data[corr_list].corr(method='kendall')
# print(total_cor)
total_corr.style.background_gradient(cmap='coolwarm')
mask = np.zeros_like(total_corr)
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
# cmaps = OrderedDict()
# plt.figure(figsize=[15, 15])
# sns.heatmap(total_corr, mask=mask, annot=True, annot_kws={"size": 10}, center=0, cmap="RdBu")
# sns.set_style("white")
# plt.savefig(f'{path}/total_kendall')
sns.pairplot(data[corr_list], kind='reg', plot_kws={'line_kws':{'color':'cyan'}}, corner=True)
plt.show()
# #use
# # correlation
corr_list = ['Year_Birth', 'Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', "Country"]
sub_corr = data[corr_list].groupby('Country').corr(method='kendall')
# # data_corr = data.columns.drop(['Dt_Customer', 'Marital_Status', 'Education', 'Country'])
# print(cor.index)
# print(type(cor))
# corr_index = list()
# for jj in nums:
#   for ii in range(0, len(data_corr)):
#     corr_index.append([jj, data_corr[ii]])
# print(cor[data_corr].index)
# print(corr_index[0:24])   ### 0:23
# corr_index = cor.index
# print(len(corr_index))
# print(cor[0:24])

# # use # # #
# # plot correlation heatmap
# sub_corr.style.background_gradient(cmap='coolwarm')
# mask = np.zeros_like(total_corr)
# triangle_indices = np.triu_indices_from(mask)
# mask[triangle_indices] = True
# for ii in range(0,8):
#   # print(0+ii*24)
#   # print(24 + 24*ii)
#   plt.figure(figsize=[15, 15])
#   sns.heatmap(sub_corr[(0 + 24 * ii):(24 + 24 * ii)], mask=mask, annot=True, annot_kws={"size": 10}, center=0, cmap="RdBu")
#   sns.set_style("white")
#   plt.title(f"{nums_sorted[ii]}", fontsize=20)
#   plt.savefig(f'{path}/{ii}_{nums_sorted[ii]}')
# plt.show()
# # #
### jointplot 
# sns.jointplot(x=x, y=y, size=5, color="blue", joint_kws={"alpha": 0.5})
# sns.jointplot(x=z, y=y, size=5, color="blue")
# plt.show()

# # plot
