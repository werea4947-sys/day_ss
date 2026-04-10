#一、多层神经网络分类实验
#将模型的各层堆叠起来，以层的方式搭建tf.keras.Sequential模型
import tensorflow as tf
mnist=tf.keras.datasets.mnist
(x_train_all,y_train_all),(x_test,y_test)=mnist.load_data()
#将Mnist数据集简单归一化
x_train_all,x_test=x_train_all/255.0,x_test/255.0
#50000个训练集，10000个验证集
x_train,x_valid=x_train_all[:50000],x_train_all[50000:]
y_train,y_valid=y_train_all[:50000],y_train_all[50000:]
print(x_train.shape)

#1．多层神经网络模型构建
import tensorflow.python.keras as keras
from tensorflow.python.keras import  models,layers,optimizers #序列模型
model=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)), # 输入层
        tf.keras.layers.Dense(256,activation=tf.nn.relu), # 隐含层1
        tf.keras.layers.Dropout(0.2),  # 20%的神经元不工作，防止过拟合
        tf.keras.layers.Dense(128,activation=tf.nn.relu),  # 隐含层2
        tf.keras.layers.Dense(64, activation=tf.nn.relu), # 隐含层3
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)  #输出层
    ])

#2．模型编译步骤
#Adam算法为训练选择优化器和确定为损失函数
model.compile(optimizer="adam",#Adam算法为训练选择优化器
                            loss="sparse_categorical_crossentropy",#损失函数采用交叉熵方法，速度更快
                            metrics=['accuracy']) #计算准确率
#输出网络参数
model.summary()

#3．模型训练
model.fit(x_train,y_train,epochs=30)

#4．模型验证
# 验证模型：
loss1,accuracy1 = model.evaluate(x_valid,y_valid,verbose=2)
loss,accuracy = model.evaluate(x_test,y_test,verbose=2)

