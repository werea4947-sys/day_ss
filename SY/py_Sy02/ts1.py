import  pandas as pd
#读取数据，确定自变量x和自变量y
data=pd.read_excel("D:\学习\python挖掘\SY\实验2 基于PYTHON的线性回归模型实验指导\发电场数据.xlsx")
x=data.iloc[:,0:4].values
y=data.iloc[:,4].values
print(x,y)
#1、2
#线性回归
#1导入线性回归模块LR
from sklearn.linear_model import  LinearRegression as  LR
#调用LR创建线性回归对象lr
lr=LR()
#fit进行拟合训练
lr.fit(x,y)
#score返回拟合优度（判定系数R²），查看线性是否显著
Slr=lr.score(x,y)
print("判定系数",Slr)
#coef_ intercept_属性，返回x对应的回归系数和回归系数常数项
c_x=lr.coef_  #回归系数
c_b=lr.intercept_     #回归系数常数项
print("x回归系数",c_x)
print("回归系数常数项",c_b)

#利用线性回归模型进行预测
#1 predict预测
import numpy as np
x1=np.array([28.4,50.6,1011.9,80.54])
x1=x1.reshape(1,4)
R1=lr.predict(x1)
print("样本预测R1",R1)
#回归方程预测
r1=x1*c_x
R2=r1.sum()+c_b
print("样本预测R2",R2)

#3
import numpy as np
x1=np.array([28.4,50.6,1011.9,80.54])
name=[["AT"],["V"],["AP"],["RH"]]
lr =LR()
# 每一列
for i in range(4):
    # 一列
    x_i = x[:, i].reshape(-1, 1)
    print(x_i)
    # Fit
    lr.fit(x_i, y)
    # R²
    Slr_i = lr.score(x_i, y)
    # 回归系数
    c_x_i = lr.coef_
    c_b_i = lr.intercept_

    name_i=name[i][0]

    # Print the results
    print(f"name {name_i}:")
    print(f"回归系数: {c_x_i[0]}")
    print(f"回归系数常数项: {c_b_i}")
    print(f"判定系数R²: {Slr_i}")
    print("\n")

