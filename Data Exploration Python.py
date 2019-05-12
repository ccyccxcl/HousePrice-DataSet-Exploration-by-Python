#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#读取数据
df_train = pd.read_csv('C:\\Users\\ccy\\Desktop\\house-prices-advanced-regression-techniques\\train.csv')
#查看数据总体情况
df_train.columns
#查看SalePrice列概况
df_train['SalePrice'].describe()
#SalePrice列直方图
sns.distplot(df_train['SalePrice']);
#plt.savefig("SalePrice列直方图.jpg")
plt.show()
#偏度和峰度（前者衡量随机变量概率分布的不对称性，后者表征概率密度分布曲线在平均值处峰值高低）
#偏度>0，均值左侧离散度比右侧弱，波形右侧长尾，偏度<0，均值左侧离散度比右侧强，波形左侧长尾
#峰度>0，波形陡峭，峰度<0，波形平缓，峰度=0，波形与博准正态分布相同
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#grlivarea/saleprice散点图
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.savefig("grlivarea-saleprice散点图.jpg")
plt.show()
#totalbsmtsf/saleprice散点图
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.savefig("totalbsmtsf-saleprice散点图.jpg")
plt.show()
#overallqual/saleprice箱型图
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.savefig("overallqual-saleprice箱型图.jpg")
plt.show()
#YearBuilt/saleprice箱型图
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
plt.savefig("YearBuilt-aleprice箱型图.jpg")
plt.show()
#相关矩阵
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.savefig("相关矩阵.jpg")
plt.show()
#saleprice相关矩阵
k = 10 #热力图变量变量个数
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.savefig("saleprice相关矩阵.jpg")
plt.show()
#散点图
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.savefig("散点图.jpg")
plt.show();
#缺失值
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#缺失值处理
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #确认没有缺失值
#数据标准化
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
#saleprice/grlivarea双变量分析
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.savefig("saleprice-grlivarea双变量分析.jpg")
plt.show()
#删除缺失值
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#TotalBsmtSF/grlivarea双变量分析
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.savefig("TotalBsmtSF-grlivarea双变量分析.jpg")
plt.show()
#直方图及正态概率图
sns.distplot(df_train['SalePrice'], fit=norm);
plt.savefig("SalePrice直方图.jpg")
plt.show()
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.savefig("SalePrice正态概率图.jpg")
plt.show()
#SalePrice数据log转化
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#SalePrice数据log化之后的直方图及正态概率图
sns.distplot(df_train['SalePrice'], fit=norm);
plt.savefig("SalePrice数据log化之后直方图.jpg")
plt.show()
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.savefig("SalePrice数据log化之后正态概率图.jpg")
plt.show()
#GrLivArea直方图及正态概率图
sns.distplot(df_train['GrLivArea'], fit=norm);
plt.savefig("GrLivArea直方图.jpg")
plt.show()
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.savefig("GrLivArea正态概率图.jpg")
plt.show()
#GrLivArea数据log转化
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#GrLivArea数据log化之后的直方图及正态概率图
plt.savefig("GrLivArea数据log化之后直方图.jpg")
plt.show()
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.savefig("GrLivArea数据log化之后正态概率图.jpg")
plt.show()
#TotalBsmtSF直方图及正态概率图
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
plt.savefig("TotalBsmtSF直方图.jpg")
plt.show()
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
plt.savefig("TotalBsmtSF正态概率图.jpg")
plt.show()
#建一个新列 (一个足够了，满足二分类特征)
#如果area>0，赋值 1, 如果area==0，赋值0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#TotalBsmtSF数据log化
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#TotalBsmtSF数据log化之后的直方图及正态概率图
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
plt.savefig("TotalBsmtSF数据log化之后直方图.jpg")
plt.show()
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.savefig("TotalBsmtSF数据log化之后正态概率图.jpg")
plt.show()
#GrLivArea/SalePrice散点图
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.savefig("GrLivArea-SalePrice散点图.jpg")
plt.show()
#TotalBsmtSF/SalePrice散点图
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
plt.savefig("TotalBsmtSF-SalePrice散点图.jpg")
plt.show()
#类别变量转化为哑变量
df_train = pd.get_dummies(df_train)
