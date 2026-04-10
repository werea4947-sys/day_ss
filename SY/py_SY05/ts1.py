# RNN应用案例
#1.IMDB数据集
#加载模块
from keras.preprocessing import sequence
from keras.models import  Sequential
from  keras.layers import Dense,Dropout,Embedding,LSTM,Bidirectional
from keras.datasets import imdb
import tensorflow as tf
#词汇表收录的单词数
max_features=10000
#加载数据
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)

#固定长度，使IMDB数据集的电影评论长度相同，pad_sequences
#一个句子长度
maxlen=100
#一个批次数据量大小
batch_size=32
#RNN输入长度固定
x_train=tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)

#RNN模型构建
#Embedding将单词编码为向量
model=Sequential()
#嵌入层
model.add(Embedding(max_features,#词汇表大小中收录单词数量，即嵌入层矩阵的行数
                    128,#每个单词的维度，即嵌入层矩阵的列数
                    input_length=maxlen))#一篇文本的长度
#搭建RNN
#定义LSTM隐含层
model.add(LSTM(128,dropout=0.25,recurrent_dropout=0.2))
#模型输出层
model.add(Dense(1,activation="sigmoid"))

#3.模型编译步骤
#模型编译
model.compile(loss='binary_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
model.summary()

#4.模型训练
model.fit(x_train,y_train,
          batch_size=batch_size,#遍历一遍数据集的批次数=len(x_train)/batch_size
          epochs=10,#遍历5遍
          validation_data=[x_train,y_train])#验证集

#5.模型验证
results=model.evaluate(x_test,y_test)
print(results)
#画图表示
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import plot_model
plot_model(model,to_file='RNN-IMDB.png',show_shapes=True)
RI=mpimg.imread('RNN-IMDB.png')
plt.imshow(RI)#显示图片
plt.axis('off')#不显示坐标轴
plt.show()

