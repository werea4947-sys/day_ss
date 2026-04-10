item =['西红柿','排骨','鸡蛋','茄子','袜子','酸奶','土豆','鞋子']
import pandas as pd
import numpy as np
data = pd.read_excel ("D:/学习/python挖掘/SY/实验6/实验6/tr.xlsx", header = None )
data = data.iloc [:,1:]
D = dict()
for t in range(len(item)):
    z = np.zeros((len(data)))
    li = list()
    for k in range (len( data.iloc [0,:])):
        s = data.iloc [:, k ]==item[t]
        li.extend(list(s[ s.values == True ].index ))
    z [li]=1
    D.setdefault(item[t], z)
Data = pd.DataFrame (D)#布尔值

# 获取字段名称，并转化为列表
c = list(Data.columns)
c0 = 0.5  # 最小置信度
s0 = 0.2  # 最小支持度
list1 = []  # 预定义列表list1,用于存放规则
list2 = []  # 预定义列表list1,用于存放规则的支持度
list3 = []  # 预定义列表list1,用于存放规则的置信度
for k in range(len(c)):
    for q in range(len(c)):
        # 对第 c[k］个项与第 c[q]个项挖掘关联规则
        # 规则的前件为c[k]
        # 规则的后件为c[q]
        # 要求前件和后件不相等
        if c[k] != c[q]:
            c1 = Data[c[k]]
            c2 = Data[c[q]]
            I1 = c1.values == 1
            I2 = c2.values == 1
            t12 = np.zeros((len(c1)))
            t1 = np.zeros((len(c1)))
            t12[I1 & I2] = 1
            t1[I1] = 1
            sp = sum(t12) / len(c1)  # 支持度
            co = sum(t12) / sum(t1)  # 置信度
            # 取置信度大于等于cO的关联规则
            if co >= c0 and sp >= s0:
                list1.append(c[k] + '--' + c[q])
                list2.append(sp)
                list3.append(co)
# 定义字典，用于存放关联规则及其置信度、支持度
R = {'rule': list1, 'support': list2, 'confidence': list3}
# 将字典转化为数据框
R = pd.DataFrame(R)
R.to_excel('/学习/python挖掘/SY/实验6/实验6/rule1.xlsx')