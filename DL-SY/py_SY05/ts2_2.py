#卷积神经网络分类识别模型：彩图
#1.数据处理
import numpy as np
import os
from PIL import Image

file="D:\学习\python挖掘\SY\实验5\实验5\水色图像"
d=os.listdir(file)#文件夹所有图片文件名
X=np.zeros((len(d),100,100,3))#预定义输入数据
Y=np.zeros(len(d))#预定义输出数据
for i in range(len(d)):
    img=Image.open(file+'\\'+d[i])#读取第i张图片,RGB
    im=img.split()
    R=np.array(im[0])#R
    row_1=int(R.shape[0]/2)-50
    row_2 = int(R.shape[0] / 2) +50
    con_1 = int(R.shape[1] / 2) - 50
    con_2 = int(R.shape[1] / 2) +50
    R=R[row_1:row_2,con_1:con_2]
    G = np.array(im[1])  # G
    G = G[row_1:row_2, con_1:con_2]
    B = np.array(im[2])  # B
    B = B[row_1:row_2, con_1:con_2]
    #取RGB三通道可，#归一化
    X[i,:,:,0]=R/255
    X[i,:,:,1]=G/255
    X[i,:,:,2]=B/255

       #构造输出数据，水色类别编号
    s=d[i]
    I=s.find('_',0,len(s))
    if int(s[:I])==1:
        Y[i]=0
    elif int(s[:I]) == 2:
        Y[i] = 1
    elif int(s[:I]) == 3:
        Y[i] = 2
    elif int(s[:I]) == 4:
        Y[i] = 3
    else:
        Y[i]=4

#划分训练集(0.8)和测试集(0.2)
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=4)

#模型实现
from keras import layers,models
#＃构建堆叠模型
model=models.Sequential()
#第一个卷积层，卷积神经元个数为32，卷积核大小为3*，默认可省
model.add ( layers.Conv2D(64,(3,3), strides =(1,1), activation ='relu',input_shape=(100,100,3)))
#紧接着的第一个池化层，2*2池化，步长为2，默认可省
model.add ( layers.MaxPooling2D((2,2), strides =2))
#第二个卷积层
model.add ( layers.Conv2D(64,(3,3), activation ='relu',))
model.add ( layers.Conv2D(64,(3,3), activation ='relu'))
#第二个池化层
model.add ( layers.MaxPooling2D((2,2)))
#第三个卷积层
model.add ( layers.Conv2D(64,(3,3), activation ='relu'))
model.add ( layers.Conv2D(64,(3,3), activation ='relu'))
#展平
model.add ( layers.Flatten ())
#全连接层
model.add ( layers.Dense (64, activation ='relu'))
model.add ( layers.Dense (64, activation ='relu'))
#输出层
model.add ( layers.Dense (5, activation ='softmax'))
#打印获得模型信息
model.summary ()

#模型优化器，损失函数，评估方法
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#评估
model.fit(x_train,y_train,epochs=550)
model.evaluate(x_test,y_test,verbose=2)

#预测
yy=model.predict(x_test)#预测结果矩阵
y1=np.argmax(yy,axis=1)#最终预测结果，取概率最大的类标签
r=y1-y_test#预测结果和实际结果
rv=len(r[r==0])/len(r)#准确率
print('预测准确率',rv)