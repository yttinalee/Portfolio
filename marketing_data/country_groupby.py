from ast import Num
from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import RegressionResults
from describtion_func import describes, total_describes
path = 'D:/Code_practice/project/MA_1'
raw_data = (f'{path}/marketing_data.csv')
data = pd.read_csv(raw_data)

nums = set(data["Country"])
nums_sorted = sorted(list(nums))
country_value_counts = data["Country"].value_counts()
lists = ["Country"]

'''Print Four Type Basic Information'''
describes(datas=data, lists=lists)


nums_sorted = sorted(list(nums))
nums_sorted_drop = nums_sorted.remove("ME")
use = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
data_groupby_home = data.groupby(by=["Country", "Kidhome"])[use]   ##, as_index=False
countis = data_groupby_home.count()
countis_sum = countis.sum(axis=1)
num_0 = countis.iloc[countis.index.get_level_values('Kidhome') == 0].sum(axis=1)
num_1 = countis.iloc[countis.index.get_level_values('Kidhome') == 1].sum(axis=1)
num_2 = countis.iloc[countis.index.get_level_values('Kidhome') == 2].sum(axis=1)

'''Plot (Types)'''
countis_sum.unstack().plot(kind="bar")  ### stacked=True
plt.title("Total response to campaign")
plt.ylabel("Total response")
plt.tight_layout()
plt.savefig(f'{path}/Kidhome-country_Campaign.png', transparent = True)

countis_sum.unstack().plot(kind="bar", stacked=True)  ###
plt.title("Total response to campaign")
plt.ylabel("Total response")
plt.tight_layout()
# plt.savefig(f'{path}/Kidhome-country_Campaign_1.png', transparent = True)

plt.show()