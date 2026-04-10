import apriori
import pandas as pd
import numpy as np
data=pd.read_excel("D:/学习/python挖掘/SY/实验6/实验6/tr.xlsx", engine='openpyxl')
D={}
item = ['西红柿', '排骨', '鸡蛋', '茄子', '袜子', '酸奶', '土豆', '鞋子']
for t in range(len(item)):
    z = np.zeros((len(data)))
    li = []
    for k in range(len(data.iloc[0, :])):
        s = data.iloc[:, k] == item[t]
        li.extend(list(s[s.values == True].index))
    z[li] = 1
    D.setdefault(item[t], z)
Data = pd.DataFrame(D)

outputfile='apriori_rules.xlsx'
support=0.2
confidence=0.4
ms='---'
apriori.find_rule(Data, support, confidence, ms).to_excel(outputfile)