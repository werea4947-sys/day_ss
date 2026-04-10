#1.数据预处理过程
#加载必要 Pandas 的模块
import pandas as pd
#读取文本数据
data = pd.read_csv ("D:\学习\python挖掘\SY\实验5\实验5\weibo_senti_100k.csv")
data = data.dropna()
#去除数据集的空值
print(data.shape)
#输出数据结构
print(data.head())#输出文本数据集的前5行

#分词
import jieba
data['data_cut']=data['review'].apply(lambda x:list(jieba.cut(x)))
#内嵌自定义函数来分词
print(data['data_cut'].head())

#去停用词
with open("D:\学习\python挖掘\SY\实验5\实验5\stopword.txt",'r',encoding='utf-8') as f:
    stop =f.readlines()
import re
stop=[re.sub('|\n|\ufeff','',r) for r in stop]
data ['data_after']=[[i for i in s if i not in stop] for s in data['data_cut']]
print(data.head())

#词向量
w=[]
for i in data['data_after']:
    w.extend(i)#将所有词语整合在一起
num_data = pd.DataFrame(pd.Series(w).value_counts())#计算出所有词语的个数
num_data['id']=list(range(1,len(num_data)+1))#把这些数据增加序号
#转化成数字
a=lambda x:list(num_data['id'][x])#以序号为序定义实现函数
data['vec']=data['data_after'].apply(a)#apply（）方法实现
print(data.head())

#构建词云
from wordcloud import  WordCloud
import matplotlib.pyplot as plt
#重组词组
num_words=[''.join(i) for i in data['data_after']]
num_words=''.join(num_words)
num_words=re.sub(' ','',num_words)
#计算全部词频
num=pd.Series(jieba.lcut(num_words)).value_counts()
#用wordcloud画图
wc_pic=WordCloud(background_color='white',font_path=r'C:\Windows\Fonts\simhei.ttf').fit_words(num)
plt.figure(figsize=(10,10))#定义图片大小
plt.imshow(wc_pic)#输出图片
plt.axis('off')#不显示坐标轴
plt.show()

#划分数据集
from sklearn.model_selection import  train_test_split
from keras.preprocessing import sequence
#一个句子长度
maxlen=100
vec_data=list(sequence.pad_sequences(data['vec'],maxlen=maxlen))
x,xt,y,yt=train_test_split(vec_data,data['label'],test_size=0.2,random_state=123)#二八
#转换数据类型
import numpy as np
x=np.array(list(x))
y=np.array(list(y))
xt=np.array(list(xt))
yt=np.array(list(yt))
print(x)

#SVM
from sklearn.svm import SVC
clf=SVC(C=1,kernel='linear')#线性
clf.fit(x,y)#模型训练
#调用报告
from sklearn.metrics import classification_report
test_pre=clf.predict(xt)#模型预测
report=classification_report(yt,test_pre)#预测结果
print("SVM",report)

#LSTM
from keras.models import  Sequential
from  keras.layers import Dense,Activation,Embedding,LSTM,Bidirectional
#模型构建
model=Sequential()
#嵌入层
model.add(Embedding(len(num_data['id'])+1, 256))#输入层，词向量表示层
model.add(Dense(64,activation="sigmoid",input_dim=100))#全连接层，32层
model.add(LSTM(128))#LSTM
model.add(Dense(1))#全连接层——输出层
model.add(Activation("sigmoid"))
model.summary()

#模型的画图表示
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import plot_model
plot_model(model,to_file='Lstm.png',show_shapes=True)
ls=mpimg.imread('Lstm.png')
plt.imshow(ls)#显示图片
plt.axis('off')#不显示坐标轴
plt.show()
model.compile(loss='binary_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
#训练模型
model.fit(x,y,validation_data=(x,y),epochs=15)
#模型验证
loss,accuracy=model.evaluate(xt,yt,batch_size=12)
print('Test loss:',loss)
print('Test accuracy:',accuracy)