#from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
#导入数据集
boston= np.load("D:/学习/python挖掘/SY/boson_data.npy", allow_pickle=True).item()
#print(boston)
X=boston["data"]
#print(X)
y=boston["target"]
#print(y)

#归一化和标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
#print(X)

#将数据集划分为训练集（60%）、验证集（20%）、测试集（20%）
import numpy as np

'''# 打乱数据集，避免因特征分布不均，导致训练不具有代表性
np.random.seed(42)  # 设置随机种子以保证可重复性
rad = np.random.permutation(len(X))#使用np.random.permutation函数对len(X)的范围进行随机排列，得到一个随机的索引顺序
X_rad = X[rad]#按照rad打乱的索引顺序重新排列X
y_rad = y[rad]#按照rad打乱的索引顺序重新排列y

# 划分的索引和占比
ratio1,ratio2=0.6,0.2
len1=len(X)
#使用int避免选择数据时，出现不是整数的情况
len_train = int(len1 * ratio1)
len_verify = int(len1 * (ratio1+ratio2))

# 切片划分数据集
X_train = X_rad[0:len_train, :]
y_train = y_rad[0:len_train]     #  y为一维数组
X_verify = X_rad[len_train:len_verify, :]
y_verify = y_rad[len_train:len_verify]
X_test = X_rad[len_verify:, :]
y_test = y_rad[len_verify:]
'''
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
# 现在，X_temp 和 y_temp 代表原始数据集的 40%
X_verify, X_test, y_verify, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("train\n",y_train)
print("verify\n",y_verify)
print("test\n",y_test)

#创建 MLPRegressor 模型
from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(solver='adam',
                   alpha=1e-5,
                   hidden_layer_sizes=(63,20,),
                   max_iter=33,
                   learning_rate_init=0.1,
                   verbose=1,
                   random_state=1,
                   activation="relu"
                   )
clf.fit(X_train,y_train)
rv=clf.score(X_train,y_train)
print("拟合优度是：",rv)
# 在验证集上进行预测
rs=clf.score(X_verify,y_verify)
print("验证精度是：",rs)
# 在测试集上进行预测
ra=clf.score(X_test,y_test)
print("测试精度是：",ra)
'''R2= clf.predict(X_verify)
# 计算均方误差
mse = mean_squared_error(y_verify,R2)
print("均方误差 (MSE):", mse)
# 计算 R^2 分数
r2 = r2_score(y_verify, R2)
print("R^2 分数:", r2)'''

from sklearn.linear_model import LinearRegression as LR
# 线性回归模型
lr = LR() #实例化一个线性回归对象
lr.fit(X_train, y_train) #采用fit方法，拟合回归系数和截距
c_b=lr.intercept_  #输出截距
c_x=lr.coef_  #输出系数   可分析特征的重要性以及与目标的关系
Slr=lr.score(X_train,y_train) #判定系数R……2
print("线性回归Slr：",Slr)
Slra=lr.score(X_test,y_test)
print("拟合优度是：",Slra)
