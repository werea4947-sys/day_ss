#卷积神经网络分类识别模型：灰图
#1.数据处理
import numpy as np
import os
from PIL import Image

file="D:\学习\python挖掘\SY\实验5\实验5\水色图像"
d=os.listdir(file)#文件夹所有图片文件名
X=np.zeros((len(d),100,100))#预定义输入数据
Y=np.zeros(len(d))#预定义输出数据
for i in range(len(d)):
    img=Image.open(file+'\\'+d[i])#读取第i张图片
    img=img.convert('L')#灰度化
    td=np.array(img)#转换为数值数组
    #或的图像中心点100*100像素的索引范围
    row_1=int(td.shape[0]/2)-50
    row_2 = int(td.shape[0] / 2) +50
    con_1 = int(td.shape[1] / 2) - 50
    con_2 = int(td.shape[1] / 2) +50
    td=td[row_1:row_2,con_1:con_2]
    X[i]=td/255#归一化

    #构造输出数据，水色类别编号
    filename=d[i]
    I=filename.find('_',0,len(filename))
    if int(filename[:I])==1:
        Y[i]=0
    elif int(filename[:I]) == 2:
        Y[i] = 1
    elif int(filename[:I]) == 3:
        Y[i] = 2
    elif int(filename[:I]) == 4:
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
#设置输入形态
model.add ( layers.Reshape ((100,100,1), input_shape =(100,100)))
#第一个卷积层，卷积神经元个数为32，卷积核大小为3*3，默认可省
model.add ( layers.Conv2D(64,(3,3), strides =(1,1), activation ='relu',padding='same'))
#紧接着的第一个池化层，2*2池化，步长为2，默认可省
model.add ( layers.MaxPooling2D((2,2), strides =2))
#第二个卷积层
model.add ( layers.Conv2D(64,(3,3), activation ='relu',padding='same'))
model.add ( layers.Conv2D(64,(3,3), activation ='relu',padding='same'))
#第二个池化层
model.add ( layers.MaxPooling2D((2,2)))
#第三个卷积层
model.add ( layers.Conv2D(64,(3,3), activation ='relu',padding='same'))
model.add ( layers.Conv2D(64,(3,3), activation ='relu',padding='same'))
#展平
model.add ( layers. Flatten ())
#全连接层
model.add ( layers.Dense (128, activation ='relu'))
model.add ( layers.Dense (128, activation ='relu'))
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
model.fit(x_train,y_train,epochs=450)
model.evaluate(x_test,y_test,verbose=2)

#预测
yy=model.predict(x_test)#预测结果矩阵
y1=np.argmax(yy,axis=1)#最终预测结果，取概率最大的类标签
r=y1-y_test#预测结果和实际结果
rv=len(r[r==0])/len(r)#准确率
print('预测准确率',rv)