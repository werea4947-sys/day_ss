S='''hello world!'''
a=S[1:7:2]
print(a)
import numpy as np
D=np.array([[1,2,np.nan,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
Dt1=D[D[:,0]>5,:]
Dt2=D[D[:,0]>5,[2,3]]
Dt3=D[[1,3],[2,3]]
print(Dt1,Dt2,Dt3)
TF=[True,False,False,True]
dt4=D[TF,:]
Dt5=D[TF,[2,3]]
d2=D[D>4]
print(dt4,Dt5,d2)
import pandas as pd
df=pd.DataFrame(D)
print(df)
ss=df.dropna(axis=1)
print(ss)
h4=df.drop(0,axis=1);
print(h4)