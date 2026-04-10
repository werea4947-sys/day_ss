#K—均值
#数据获取及标准化处理
import pandas as pd
data=pd.read_excel("D:\学习\python挖掘\SY\实验6\实验6\农村居民人均可支配收入来源2016.xlsx")
X=data.iloc[:,1:]
from sklearn.preprocessing import  StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

#K-均值聚类分析
#（1）导入K - 均值聚类模块KMeans。
from sklearn.cluster import KMeans
#（2）利用KMeans创建K - 均值聚类对象model。
model=KMeans(n_clusters= 4,random_state=0,max_iter=500)
#（3）调用model对象中的fit()方法进行拟合训练。
model.fit(X)
#（4）获取model对象中的labels_属性，可以返回其聚类的标签。
c=model.labels_
Fs=pd.Series(c,index=data['地区'])
Fs=Fs.sort_values(ascending=True)
print(Fs)