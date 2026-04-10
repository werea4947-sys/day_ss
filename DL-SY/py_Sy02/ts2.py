import pandas as pd
#获取数据
data=pd.read_excel("D:\学习\python挖掘\SY\实验2 基于PYTHON的线性回归模型实验指导\credit.xlsx")

#训练样本和测试样本划分
x=data.iloc[:600,:14].values #训练样本特征数据x
y=data.iloc[:600,14].values #训练样本预测变量y
x1=data.iloc[600:,:14].values #测试样本x1
y1=data.iloc[600:,14].values #测试样本y1

#逻辑回归分析
from sklearn.linear_model import  LogisticRegression as LR
lr=LR()
#训练
lr.fit(x,y)
#返回模型准确率
r=lr.score(x,y);
#predict预测
R=lr.predict(x1)
Z=R-y1
Rs=len(Z[Z==0])/len(Z)
print("预测结果",R)
print(f"预测准确率 {Rs:.4f}")

