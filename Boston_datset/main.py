from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
# # # plots
import matplotlib.pyplot as plt
import seaborn as sns
# # # random split data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# # #　Statistic 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import RegressionResults
boston_dataset = load_boston()
# print(type(boston_dataset))
# print(boston_dataset)
# # # list for attributes
# print(dir(boston_dataset))
# print(boston_dataset.DESCR)

## Data points and features
# print(boston_dataset.data)
# print(type(boston_dataset.data))
# print(boston_dataset.data.shape)
# print(boston_dataset.feature_names)
# print(boston_dataset.target)

### Data exploration with Pandas dataframes
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
## Add column with the target (price)
data["PRICE"] = boston_dataset.target
# print(data.head)
# print(data.tail)
# print(data.count())

## Cleaning data -- check for missing values
## find missing
# missing = pd.isnull(data).any()
# print(missing)
## data info
# print(data.info())

# # # Visualization Data -- Histograms, Disturbutions and Bar Charts
# plt.figure(figsize=[8,6])
# plt.hist(data["PRICE"], bins=50, ec="black", color="#2196F3")
# plt.xlabel("Price in 000s")
# plt.ylabel("Nr. of Houses")
# plt.show()

## import seaborn as sns
## colors https://www.materialpalette.com/yellow/amber
# plt.figure(figsize=[8,6])
# sns.distplot(data["PRICE"], bins=50, hist_kws={"color":"#FFC107"}, kde_kws={"color": "#303F9F"})
# plt.xlabel("Price in 000s")
# plt.ylabel("Nr. of Houses")
# plt.show()

## RM plot
# sns.distplot(data["RM"], hist_kws={"color": "#536DFE"}, kde_kws={"color": "#303F9F"})
# plt.xlabel("Average Number of Rooms")
# plt.ylabel("Nr. of Houses")
# plt.show()

# print(data["RM"].mean())

## RAD   ##  a rank for accessibility to highways
# print(data["RAD"].mean())
# print(data["RAD"].value_counts())
# plt.figure(figsize=[7,5])
# plt.hist(data["RAD"], bins=24, ec="#7b1fa2", color= "#303F9F", rwidth=0.5)
# # sns.distplot(data["RAD"], bins=24, hist_kws={"color": "#7b1fa2"}, kde_kws={"color": "#303F9F"})
# plt.xlabel("Accessibility to Highways")
# plt.ylabel("Nr. of Houses")
# plt.show()

## bar chart
# frequency = data["RAD"].value_counts()
# # print(type(frequency))
# ## x_index = frequency.index
# x_index = frequency.axes[0]
# plt.figure(figsize=[7,5])
# plt.bar(x_index, frequency)
# ## sns.barplot(x=x_index, y=frequency)
# plt.xlabel("Accessibility to Highways")
# plt.ylabel("Nr. of Houses")
# plt.show()

# # # Descriptive Statistics
# print(data["PRICE"].min())
# print(data["PRICE"].max())
# print(data.min())
# print(data.max())
# print(data.mean())
# print(data.median())
# # # # # # # #
# print(boston_dataset.DESCR)
# print(data.describe())
# # # # # # # #
### CHAS
# print(data["CHAS"].value_counts())
# plt.hist(data["CHAS"], bins=2, rwidth=0.5)
# plt.show()

# # # # Correlations  ## default is Person
cor1_result = data["PRICE"].corr(data["RM"])
# print("PRICE*RM",cor1_result)
# cor2_result = data["PRICE"].corr(data["PTRATIO"])
# print("PRICE*PTRATIO", cor2_result)
# print(data.corr())
## method 1 of mask
mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
# print(triangle_indices)
# print(mask)
mask[triangle_indices] = True
# print(mask)
# print(np.triu_indices(data.corr()))
## https://seaborn.pydata.org/generated/seaborn.heatmap.html
# plt.figure(figsize=[10,8])
# sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={"size": 10}, cmap="YlGnBu")
# sns.set_style("white")
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()


##method 2 of mask
# mask = np.triu_indices(14)
# data_corr = data.corr().to_numpy()
# # print(data_corr)
# # print(type(data_corr))
# data_corr[mask] = 0
# print(data_corr) 


## DIS & NOX scatter
# nox_dis_corr = round(data["NOX"].corr(data["DIS"]), 3)
# # plt.figure(figsize=[8,6])
# plt.title(f"DIS x NOX (Correlation {nox_dis_corr})")
# plt.scatter(data["DIS"],data["NOX"], alpha=0.6, s=40, color="indigo")
# plt.xlabel("DIS -  Distance from employment",fontsize=10)
# plt.ylabel("NOX - Nitric Oxide Pollution", fontsize=10)
# plt.show()

# plt.figure(figsize=[8,6])
# sns.set()
# # sns.set_context("talk")
# sns.set_style("whitegrid")
# ### jointplot ### http://seaborn.pydata.org/generated/seaborn.jointplot.html
# # sns.jointplot(x=data["DIS"], y=data["NOX"], size=5, color="blue", joint_kws={"alpha": 0.5})
# sns.jointplot(x=data["DIS"], y=data["NOX"], size=5, color="blue", kind="hex", height=15)
# plt.show()

### TAX & RAD
# tax_dis_rad

# sns.set()
# sns.set_context("talk")
# sns.set_style("whitegrid")
### jointplot ### http://seaborn.pydata.org/generated/seaborn.jointplot.html
# sns.jointplot(x=data["DIS"], y=data["NOX"], size=5, color="blue", joint_kws={"alpha": 0.5})
# sns.jointplot(x=data["TAX"], y=data["RAD"], size=5, color="blue")
# plt.show()
## draw correlation line
# sns.set()
# sns.lmplot(x="TAX", y="RAD", data=data, size=5)
# plt.show()

# rm_price_corr = round(data["RM"].corr(data["PRICE"]),3)
# plt.figure(figsize=[9,6])
# plt.scatter(x=data["RM"], y=data['PRICE'], alpha=0.6, s=50, color="#303F9F")
# plt.title(f"RM vs PIRCE (Correlation {rm_price_corr})", fontsize=12)
# plt.xlabel("Average rooms")
# plt.ylabel("House Price")
# plt.show()

# sns.lmplot(x="RM", y="PRICE", data=data, size=7)
# plt.gcf().subplots_adjust(left=0.1, bottom=0.1)
# plt.show()
# # #　# # #　# # #　# # #　# # #　 Multivariable Regression # # #　# # #　# # #　# # #　# # #　# # #　
##### plot all the features correlations
#### Warning it will took quite long time(~seconds)
##sns.pairplot(data, corner=True)
# sns.pairplot(data, kind="reg", plot_kws={"line_kws": {"color": "cyan"}}, corner=True)
# plt.gcf().subplots_adjust(left=0.1, bottom=0.1)
# plt.show()

# # #　# # #　# # #　Split Training & Testing Data　# # #　# # #　# # #　
prices = data["PRICE"]
features = data.drop("PRICE", axis=1)  ###  features not include PRICE
# print(data.head())
# print(features)
# # tuple unpacking
x_train, x_test, y_train, y_test = train_test_split(features, prices, 
                                                test_size=0.2, random_state=10)
## % of training set
# print(len(x_train)/ len(features))
## % of test data set
# print(x_test.shape[0]/features.shape[0])

regr = LinearRegression()
# # # r for training data
regr.fit(x_train, y_train)
# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　
### r-squared training data ###
# print("Training data r-squared:", regr.score(x_train, y_train))
# print("Test data r-squared:", regr.score(x_test, y_test))
# ###
# print("Intercept of training data", regr.intercept_)
coef_result = pd.DataFrame(data=regr.coef_, index=x_train.columns, columns=["coef"])
# print(coef_result)
# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #
# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　

# # #　# # #　# # #　# # #　# # #　Data Transformations # # #　# # #　# # #　# # #　# # #　# # #　
# print(data["PRICE"].skew())
# y_log = np.log(data["PRICE"])
# print(y_log.tail())
# print(y_log.skew())
# sns.distplot(y_log)
# plt.title(f"Log price with skew {y_log.skew()}")
# plt.show()

# sns.lmplot(x="LSTAT", y="PRICE", data=data, size=6, scatter_kws={"alpha": 0.6}, line_kws={"color": "darkred"})
# plt.gcf().subplots_adjust(left=0.1, bottom=0.1)
# # plt.show()

# transformed_data = features
# transformed_data["LOG_PRICE"] = y_log
# sns.lmplot(x="LSTAT", y="LOG_PRICE", data=transformed_data, size=6, scatter_kws={"alpha": 0.6}, line_kws={"color": "cyan"})
# plt.gcf().subplots_adjust(left=0.1, bottom=0.1)
# plt.show()
# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　# # #　
# # #　# # #　# # #　# # #　# # #　# # #　Regession using LOG Transformation version# # #　# # #　# # #　# # #　# # #　# # #　
# # #　datas
# prices_log = np.log(data["PRICE"])
# features = data.drop(["PRICE"], axis=1)
# # # #　split data
# x_train, x_test, y_log_train, y_log_test = train_test_split(features, prices_log, test_size=0.2, random_state=10)

# regr_log = LinearRegression()
# regr_log.fit(x_train, y_log_train)

# print("Training data r-squared:", regr.score(x_train, y_log_train))
# # print("Test data r_squreded:", regr.score(x_test, y_log_test))
# # print("Intercept", regr_log.intercept_)

# coef_log_result = pd.DataFrame(data=regr_log.coef_, index=x_train.columns, columns=["coef"])
# print(coef_log_result)

# ## Charles River Property Premium
# # # Get reverse transformation to raw price # # #
# print(np.e**coef_log_result["coef"]["CHAS"])
# # print("CHAS",coef_log_result["coef"]["CHAS"])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # #  p-values & Evaluating Coefficient # # # # # # # # # # # # # # # # # # 
# # # import statsmodels.api as sm
x_incl_const = sm.add_constant(x_train)
model = sm.OLS(y_train, x_incl_const)   # # # (target, training_data_intercept)
results = model.fit()
# print("results.params\n",results.params)
# print("results.pvalues\n",results.pvalues)
# t_test_result = results.t_test([13,0])
# print(t_test_result)
pd_results = pd.DataFrame({'coef': results.params, 'p-values': round(results.pvalues,3)})
# print("pd_results\n",pd_results)
# print(results.bic)
# # # # # # # # # Testin for Multicollinearity # # # # # # # # # 
# # # from statsmodels.stats.outliers_influence import variance_inflation_factor
variance_inflation_factor(exog=x_incl_const.values, exog_idx=1)
# print(type(x_incl_const))
# print(x_incl_const.columns)
# print(x_incl_const.shape[1])
# num = x_incl_const.columns.get_loc("TAX")
# print(num)
# vif = []
# for i in range(x_incl_const.shape[1]):
#   vif.append(variance_inflation_factor(exog=x_incl_const.values, exog_idx=i))
vif = [variance_inflation_factor(exog=x_incl_const.values, exog_idx=i) for i in range(x_incl_const.shape[1])]
pd_vif = pd.DataFrame({"coef_name": x_incl_const.columns, "vif": np.around(vif, 2)})
# print(pd_vif)
# print(pd_vif.loc[1])
# print(pd_vif["vif"][1])
# pd_vif = pd.DataFrame({"vif": np.around(vif, 2)},index=x_incl_const.columns)
# print(pd_vif)
# print(pd_vif.loc["CRIM"])
# print(pd_vif["vif"]["CRIM"])
# print(pd_vif["vif"][1])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # Simplify the Model : transformed (using log prizes# & simplified (dropping featrues) # # # 
# # # # Model Simplification & the BIC # # # # 
# # Original model with log prices and all features # #
'''Find & check official docs for results object & print out BIC & r-squared '''
''' BIC = n*log(residual sum of squares/n) + k*log(n) '''
regr_log = LinearRegression()
prices_log = np.log(data["PRICE"])
features = data.drop(['PRICE'], axis=1)
x_train, x_test, y_log_train, y_log_test = train_test_split(features, prices_log, test_size=0.2, random_state=10)


''' log with INDUS '''
print("log with INDUS")
x_incl_const = sm.add_constant(x_train)
model = sm.OLS(y_log_train, x_incl_const)   # # # (target, training_data_intercept)
results = model.fit()

org_coef = pd.DataFrame({'coef': results.params, 'p-values': round(results.pvalues,3)})
# print("org_coef:\n",org_coef)
print("BIC:", results.bic)
print("R-squared:", results.rsquared)
''' log without INDUS '''
print("\n\nlog without INDUS")

x_incl_const_b = x_incl_const.drop(["INDUS"], axis=1)
model_b = sm.OLS(y_log_train, x_incl_const_b)
results_b = model_b.fit()
# print(results_b)
org_coef_b = pd.DataFrame({"coef": results_b.params, "p-values": round(results_b.pvalues,3)})
# print("org_coef_b:\n",org_coef_b)
print("BIC:",results_b.bic)
print("R-squared:", results_b.rsquared)

''' Reduced model # excluding INDUS & AGE '''
print("\n\nlog without INDUS & AGE")
x_incl_cont_c = x_incl_const.drop(["INDUS","AGE"],axis=1)
model_c = sm.OLS(y_log_train, x_incl_cont_c)
results_c = model_c.fit()
org_coef_c = pd.DataFrame({"coef": results_c.params, "p-values": round(results_c.pvalues, 3)})
# print("c - coef:\n",org_coef_c)
print("BIC: ", results_c.bic)
print("R_squared: ", results_c.rsquared)


# frames_coef = [org_coef, org_coef_b, org_coef_c]
# print(pd.concat(frames_coef, axis=1))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # #　Folloe the simplified models # # # Residuals # # #
print("log with INDUS")
residuals = y_log_train - results.fittedvalues
# print(type(residuals))
# print(residuals)
# print(residuals.describe())
# print(results.resid)
## Graph  of Axtual vs. Predicted Prices
log_corr = round(y_log_train.corr(results.fittedvalues), 2)
# print(log_corr)
plt.figure(figsize=[15,5])
plt.subplot(131)
plt.scatter(x=y_log_train, y=results.fittedvalues, c="navy", alpha=0.6)
plt.plot(y_log_train, y_log_train, color="cyan")
plt.xlabel("Actual log prices $y _i$", fontsize=10)
plt.ylabel("Predicted log prizes $\hat y _i$", fontsize=10)
plt.title(f"Actual vs. Predicted log prizes:\n $y _i$ vs. $y _i$(Corr: {log_corr})",fontsize=15)
# # plt.show()
# plt.figure()
plt.subplot(132)
plt.scatter(x=np.e**y_log_train, y=np.e**results.fittedvalues, c="blue", alpha=0.6)
plt.plot(np.e**y_log_train, np.e**y_log_train, color="cyan")
plt.xlabel("Actual log prices 000s $y _i$", fontsize=10)
plt.ylabel("Predicted log prizes $\hat y _i$", fontsize=10)
plt.title(f"Actual vs Predicted log prizes:\n $y _i$ vs. $y _i$(Corr: {log_corr})",fontsize=15)
# # plt.show()

# ##### Residuals vs. Predicted values( y_hat, residuals)
# plt.figure()
plt.subplot(133)
plt.scatter(x=results.fittedvalues, y=results.resid, c="blue", alpha=0.6)
# plt.plot(np.e**y_log_train, np.e**y_log_train, color="cyan")
plt.xlabel("Actual log prices 000s $y _i$", fontsize=10)
plt.ylabel("Residuals", fontsize=10)
plt.title(f"Residuals vs Predicted Values {log_corr})",fontsize=15)
plt.gcf().subplots_adjust(left=0.1, bottom=0.1)
# # plt.show()

### Mean squared Error $ R-squared
reduced_log_mse = round(results.mse_resid, 3)
reduced_log_rsquared = round(results.rsquared, 3)
# # # # # # # # # # # # # Disturbution of Residuals (log prices) - checking for normality # # # # # # # # # # # # 
resid_mean = round(results.resid.mean(),3)
resid_skew = round(results.resid.skew(), 3)
plt.figure(figsize=[8,5])
sns.distplot(results.resid, color="navy")
plt.title(f"Log price model: Residuals Mean: ({resid_mean}), Skew: {resid_skew}")
# plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # Challenge # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # Using the original model with all the features and normal prices generte # # # # # # # # 
# # # # # # # # without LOG # # # # # # # # 
''' Plot of actual vs predicted prices (incl. correlation) using a different colour '''
''' Plot of actual vs predicted prices '''
''' Plot of disturbution of residuals (incl. skew)'''
''' Analyze the results'''
data["PRICE"] = boston_dataset.target

# # #　# # #　# # #　Split Training & Testing Data　# # #　# # #　# # #　
prices = data["PRICE"]
features = data.drop("PRICE", axis=1)
#### from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)
regr = LinearRegression()
regr.fit(x_train, y_train)
coef_result = pd.DataFrame(data=regr.coef_, index=x_train.columns, columns=["coef"])

x_incl_const = sm.add_constant(x_train)
model = sm.OLS(y_train, x_incl_const)
results = model.fit()
coef_pd_results = pd.DataFrame({"coef": results.params, "p-values": round(results.pvalues,3)})
''' VIF '''
# print(variance_inflation_factor(exog=x_incl_const.values, exog_idx=1))
vif = [variance_inflation_factor(exog=x_incl_const.values, exog_idx=i) for i in range(x_incl_const.shape[1])]
pd_vif = pd.DataFrame({"coef_name": x_incl_const.columns, "vif": np.round(vif, 2)})
plt.figure(figsize=[6, 6])
# plt.scatter(x=x_train[i],y=y_train)
# sns.pairplot(data, kind="reg", plot_kws={"line_kws":{"color":"cyan"}}, corner=True)

plt.scatter(x=y_train, y=results.fittedvalues, c="purple",alpha=0.5)
plt.plot(y_train, y_train, color="#303F9F")
plt.xlabel("Actual prices 000s $y _i$", fontsize=10)
plt.ylabel("Predicted prices 000s $\hat y _i$", fontsize=10)
plt.title("Actual vs Predicted prices: $y _i$ vs $\hat y _i$", fontsize=14)
plt.figure(figsize=[6, 6])

plt.scatter(x=results.fittedvalues, y=results.resid, c="blue", alpha=0.5)
plt.xlabel("Predicted prices $\hat y _i$", fontsize=10)
plt.ylabel("Residuals", fontsize=10)
plt.title("Residuals vs Fitted Values", fontsize=10)
plt.figure(figsize=[6, 6])

resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)
sns.distplot(results.resid, color='indigo')
plt.title(f"Residuals Skew: {resid_skew}, Mean: {resid_mean}")
plt.show()
### Mean squared Error $ R-squared
full_normall_mse = round(results.mse_resid, 3)
full_normall_rsquared = round(results.rsquared, 3)
