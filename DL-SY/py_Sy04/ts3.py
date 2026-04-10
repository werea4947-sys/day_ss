#三、 卷积神经网络（CIFAR-10 数据集）
#1．CIFAR-10 数据集
import tensorflow as tf
from keras import  models,layers,datasets
import  matplotlib.pyplot as plt

#数据加载
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
print(train_images.shape,' ',train_labels.shape) #看看数据集情况
#数据集简单归一化
train_images,test_images=train_images/255.0,test_images/255.0
#数据集的类型
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#画出数据集的大概预览
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

#2．CNN 模型构建
#CNN模型构建
model=models.Sequential()
#卷积层
#input_shape表示卷积层输入，filter表示卷积核大小
#stride表示卷积步长
#padding表示控制卷积核处理边界的策略，激活函数用relu
model.add(layers.Conv2D(input_shape=(32,32,3),filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu')) #32个卷积核，卷积核大小3x3
#池化层，窗口2x2，最大池化抽样
model.add(layers.MaxPool2D(pool_size=(2,2)))
#卷积层，64个卷积核，卷积核大小3x3
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="valid",activation='relu'))
#池化层，窗口2x2
model.add(layers.MaxPool2D(pool_size=(2,2)))
#卷积层，64个卷积核，卷积核大小3x3
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="valid",activation='relu'))
#全连接层、flatten()将卷积和池化后提取的特征摊平后输入全连接网络
model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu'))
#分类层——输出10个种类分类
model.add(layers.Dense(10))

#3．模型编译步骤
#CNN模型编译
#优化器用Adam，损失函数用交叉熵
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.summary() #输出模型参数结构

#4．模型训练
#CNN模型训练
history=model.fit(
    train_images,train_labels,
    epochs=10,validation_data=(test_images,test_labels))
#hitory对象有一个history成员，它是一个字典，包含训练过程的所有数据
plt.plot(history.history['accuracy'],label=['accuracy'])
plt.plot(history.history['val_accuracy'],label=['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5,1])
plt.legend(loc='lower right')
plt.show()

#5．模型验证
#测试集测试泛化模型效果
test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
print(test_acc) #输出准确率
#画出预测图
import numpy as np
#prediction=model.predict_classes(test_images)
prediction=model.predict(test_images)
prediction=np.argmax(prediction,axis=1)
print("展示测试集第一张图片的模型识别是：")
print("%s\n"%(prediction[0]))
print("测试集第一张的实际结果是：")
print(test_labels[0])
print("展示该图片")
plt.imshow(test_images[0])
plt.show()