#SVM
#数据获取
import pandas as pd
data = pd.read_excel("D:\学习\python挖掘\SY\实验6\实验6\car.xlsx")
#样本划分
x=data.iloc[:1690,:6].values #训练样本特征数据x
y=data.iloc[:1690,6].values #训练样本预测变量y
x1=data.iloc[1691:,:6].values #测试样本x1
y1=data.iloc[1691:,6].values #测试样本y1
#SVM分类模型构建
#导入支持向量机模块SVM
from sklearn import svm
#创建类SVM
clf=svm.SVC(kernel='rbf')
#fit训练
clf.fit(x,y)
#score,模型训练精度
rv=clf.score(x,y);
#预测
R=clf.predict(x1)
#预测精度
Z=R-y1
Rs=len(Z[Z==0])/len(Z)
print('模型训练精度：',rv)
print('预测结果：',R)
print('预测准确率：',Rs)