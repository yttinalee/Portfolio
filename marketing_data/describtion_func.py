import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path = 'D:/Code_practice/project/MA_1'
raw_data = (f'{path}/marketing_data.csv')
data = pd.read_csv(raw_data)
a = 'Kidhome'
b = 'Teenhome'
lists = [a, b]

products_list = ['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
channels_list = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
basic_info_list = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome','Teenhome', 'Dt_Customer', 'Recency',  'Response', 'Complain']
cmp_list = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
        'AcceptedCmp5']
lists_use = [products_list, basic_info_list, channels_list, cmp_list]
nums = set(data["Country"])
nums_sorted = sorted(list(nums))
def total_describes(datas, lists_use=lists_use):
  for kk in lists_use:
    print(data[kk].describe())

def describes(datas, lists, lists_use=lists_use, sorts=False):
  for ll in lists:
    print("\n\n",ll)
    data_groupby = data.groupby(by=[ll])
    value_counts = data[ll].value_counts()
    print(value_counts)
    for kk in lists_use:
      # print(kk)
      if kk != cmp_list:
        if kk == products_list:
          print("\n\nProducts")
        elif kk == basic_info_list:
          print("\n\nINFO")
          sub_list = ['Income', 'Kidhome','Teenhome']
        elif kk == channels_list:
          print("\n\nChannels") 
        groupby_drop_sum = round(data_groupby[kk].sum(),2)
        groupby_drop_mean = round(data_groupby[kk].mean(),2)
        groupby_drop_max = data_groupby[kk].max()
        groupby_drop_min = data_groupby[kk].min()
        groupby_drop_std = round(data_groupby[kk].std(),2)
        groupby_drop_percentile = data_groupby[kk].quantile(q=[0.25, 0.5, 0.75])
        print('sum\n',groupby_drop_sum)
        print('mean\n',groupby_drop_mean)
        print('max\n',groupby_drop_max)
        print('min\n',groupby_drop_min)
        print('std\n',groupby_drop_std)
        print('percentile\n',groupby_drop_percentile)
        if kk == products_list:
          groupby_drop_sumsum = groupby_drop_sum[kk].sum(axis=1)
          print(groupby_drop_sumsum)
          # with pd.ExcelWriter(f"{path}/products.xlsx") as writer:
            # groupby_drop_sum.to_excel(writer, sheet_name="sum")
            # groupby_drop_mean.to_excel(writer, sheet_name="mean")
            # groupby_drop_std.to_excel(writer, sheet_name="std")          
        # elif kk == basic_info_list:

        #   sub_list = ['Kidhome','Teenhome']
        #   groupby_drop_sumsum = groupby_drop_sum[sub_list].sum(axis=1)
        #   print(groupby_drop_sumsum) 
        elif kk == channels_list:
          sub_list = ['NumWebPurchases', 'NumStorePurchases']
          groupby_drop_sumsum = groupby_drop_sum[sub_list].sum(axis=1)
          print(groupby_drop_sumsum) 
          # with pd.ExcelWriter(f"{path}/channels.xlsx") as writer:
            # groupby_drop_sum.to_excel(writer, sheet_name="sum")
            # groupby_drop_mean.to_excel(writer, sheet_name="mean")
            # groupby_drop_std.to_excel(writer, sheet_name="std")    
        # elif kk == others:
        #   plt.figure("product _ sum")
        #   plt.bar(x=products_list, y=groupby_drop_sum)
        #   plt.title("Total Sales in Countries")
      else:
        print("\n\nCmps")
        cmp = data[kk]
        print(cmp.sum())
        groupby_drop_sum = round(data_groupby[kk].sum(),2)
        groupby_drop_sumsum = groupby_drop_sum.sum(axis=1)
        # groupby_drop_sumsum_percentage = groupby_drop_sum.sum(axis=1)/groupby_drop_sum[sub_list].count()
        print('sum\n',groupby_drop_sum)
        print('subgroup_sum\n', groupby_drop_sumsum)
        # plt.figure(1)
        # plt.bar(x=nums_sorted, height=groupby_drop_sumsum)
        # plt.xlabel("Country")
        # plt.ylabel("Total response")
        # plt.title("Total Response of campaign in Countries")
        # plt.savefig(f"{path}/cmp_country.png", transparent = True)
        

  # plt.show()
    # if data_groupby['kidhome'] == 0:
