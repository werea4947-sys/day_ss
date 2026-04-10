import matplotlib.pyplot as plt
import  numpy as np
import  pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 读取数据
data = pd.read_excel("D:\学习\python挖掘\SY\实验1-主成分分析及PYTHON综合应用-20231107\DATA.xlsx")
# 添加 ID 列
data['ID'] = range(1, len(data) + 1)
data.to_excel('data.xlsx', index=False)
# 筛选大于 0 的数据
data1 = data[data.iloc[:, 2:] > 0]
data1 = data1.iloc[:, 2:]
# 删除包含 NaN 值的行
data1 = data1.dropna(axis="index", how="any")
# 保存 data1 到 Excel
data1.to_excel('data1.xlsx', index=False)
# 使用 'ID' 列进行关联
data2 = pd.merge(data1, data, how='inner', on=['ID'])
#去除Nan和负值的excel——data2
data2.to_excel('data2.xlsx', index=False)
#定义函数
def calculate_pca_and_f_score(data, n):
    # Standardize the data
    X = data.iloc[:, 13:24]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # PCA
    pca = PCA(n_components=n)
    Y = pca.fit_transform(X)
    # 累计贡献率
    gxl = pca.explained_variance_ratio_

    # 计算 F 分数
    F = np.zeros((len(Y)))
    for i in range(len(gxl)):
        f = Y[:, i] * gxl[i]
        F = F + f

    ''''# 创建 fs1Series --index 为股票代码
    fs1 = pd.Series(F, index=data2['ts_code'].explode().values)
    # 排序 F 分数 --index 为股票代码
    Fscore1 = fs1.sort_values(ascending=False)'''

    #index 为股票中文名称
    stk = pd.read_excel(r"D:\学习\python挖掘\SY\实验1-主成分分析及PYTHON综合应用-20231107\stkcode.xlsx")
    # 合并 data2 和 stk 数据框，利用ts_code链接
    stkmerge = pd.merge(data2, stk[['ts_code', 'name']], how='left', on='ts_code')
    # 创建 Fscore2 Series
    fs2 = pd.Series(F, index=stkmerge['name'].values)
    # index 为股票中文简称
    Fscore2 = fs2.sort_values(ascending=False)


    return Fscore2

data_100= calculate_pca_and_f_score(data2, 1)
print("累计贡献率100\n",data_100)
data_95=calculate_pca_and_f_score(data2, 0.95)
print("累计贡献率95\n",data_95)
data_85=calculate_pca_and_f_score(data2, 0.85)
print("累计贡献率85\n",data_85)
data_75=calculate_pca_and_f_score(data2, 0.75)
print("累计贡献率75\n",data_75)
data_65=calculate_pca_and_f_score(data2, 0.65)
print("累计贡献率65\n",data_65)
data_50=calculate_pca_and_f_score(data2, 0.50)
print("累计贡献率50\n",data_50)


#作图
#gxl=100
plt.figure(1)
plt.subplot(2,3,1)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title("累计贡献率100")
x1 = np.arange(1, 21)
y1 = data_100.iloc[:20].values
plt.plot(x1,y1,'r*--')
plt.xticks(range(1,21),data_100.index.values[range(0,20)],rotation=45, fontsize=2)
plt.xlabel('银行名称')
plt.ylabel('得分')
plt.legend(["贡献率100"])

#95
plt.subplot(2,3,2)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title("累计贡献率95")
x1 = np.arange(1, 21)
y1 = data_95.iloc[:20].values
plt.plot(x1,y1,'*--')
plt.xticks(range(1,21),data_95.index.values[range(0,20)],rotation=45, fontsize=2)
plt.xlabel('银行名称')
plt.ylabel('得分')
plt.legend(["贡献率95"])
#85
plt.subplot(2,3,3)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title("累计贡献率85")
x1 = np.arange(1, 21)
y1 = data_85.iloc[:20].values
plt.plot(x1,y1,'r*--')
plt.xticks(range(1,21),data_85.index.values[range(0,20)],rotation=45, fontsize=2)
plt.xlabel('银行名称')
plt.ylabel('得分')
plt.legend(["贡献率85"])
#75
plt.subplot(2,3,4)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title("累计贡献率75")
x1 = np.arange(1, 21)
y1 = data_75.iloc[:20].values
plt.plot(x1,y1,'r*--')
plt.xticks(range(1,21),data_75.index.values[range(0,20)],rotation=45, fontsize=2)
plt.xlabel('银行名称')
plt.ylabel('得分')
plt.legend(["贡献率75"])
#65
plt.subplot(2,3,5)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title("累计贡献率65")
x1 = np.arange(1, 21)
y1 = data_65.iloc[:20].values
plt.plot(x1,y1,'r*--')
plt.xticks(range(1,21),data_65.index.values[range(0,20)],rotation=45, fontsize=2)
plt.xlabel('银行名称')
plt.ylabel('得分')
plt.legend(["贡献率65"])
#50
plt.subplot(2,3,6)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title("累计贡献率50")
x1 = np.arange(1, 21)
y1 = data_50.iloc[:20].values
plt.plot(x1,y1,'r*--')
plt.xticks(range(1,21),data_50.index.values[range(0,20)],rotation=45, fontsize=2)
plt.xlabel('银行名称')
plt.ylabel('得分')
plt.legend(["贡献率50"])
plt.tight_layout()
plt.savefig('gxl.png', dpi=800)
plt.show()