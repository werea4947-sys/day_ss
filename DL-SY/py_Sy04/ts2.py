#二、多层神经网络回归问题应用
#1．Auto MPG 数据集
import  matplotlib.pyplot as plt
import  pandas as pd
import seaborn as sns
import tensorflow as tf

#2．数据集清洗与划分
colium_names=['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin'] #选定需要的数据属性
raw_dataset=pd.read_csv("D:/学习/python挖掘/SY/实验4数据/auto-mpg.data",
                        names=colium_names,
                        na_values="?",
                        comment="\t",
                        sep=" ",
                        skipinitialspace=True) #读取数据集
dataset=raw_dataset.copy() #复制数据集
print(dataset.shape)
print(dataset.tail()) #查看最后5行数据
#数据清洗，数据集中包括一些缺漏值，空值等异常值
dataset.isna().sum() #判断是否有空值，并计算总数
dataset=dataset.dropna()
print(dataset.shape)
print(dataset.head())
origin=dataset.pop('Origin') #把这列取出，pop函数一处列表中元素并赋值
dataset['USA']=(origin==1)*1.0 #添加USA列，当origin为1时赋值1
dataset['Europe']=(origin==2)*1.0
dataset['Japan']=(origin==3)*1.0
print(dataset.tail()) #查看最后5行数据
#划分训练数据集和测试数据集
train_dataset=dataset.sample(frac=0.8,random_state=0) #训练集占80%
test_dataset=dataset.drop(train_dataset.index)
print(train_dataset.shape)
#查看总体数据统计
train_stats=train_dataset.describe()
train_stats.pop('MPG')
train_stats=train_stats.transpose()
print(train_stats)
#画图查看训练集
sns.pairplot(train_dataset[["MPG","Cylinders","Displacement","Weight"]],diag_kind="kde")
plt.show()
#分离特征
train_labels=train_dataset.pop("MPG")#训练集去掉MPG值
test_labels=test_dataset.pop("MPG")
#标准化处理
def norm(x):
    return (x-train_stats["mean"])/train_stats["std"] #标准化公式
normed_train_data=norm(train_dataset)
normed_test_data=norm(test_dataset)

#3．多层神经网络模型构建
#建立3层网络，结点[64,64,1],激活函数relu
def build_model():
    model=tf.keras.Sequential([
        tf.keras.layers.Dense(64,activation='relu',input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    #自定义PMSprop优化器，学习率时0.001
    optimizer=tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])#损失函数用mse
    return  model
#模型实例化
model=build_model()
model.summary()

#4．模型训练
#对模型进行100个周期的训练，并在history对象中记录训练和验证的准确性
history=model.fit(
    normed_train_data,train_labels,
    epochs=100,validation_split=0.2,verbose=0) #verbose=0表示不输出训练记录
#输出训练的各项指标值
'''hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
print(hist.tail())'''
#把训练结果用图形表示
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.xlabel('训练次数')
    plt.ylabel('平均绝对误差[MPG]')
    plt.plot(hist['epoch'],hist['mae'],label=['训练误差'])
    plt.plot(hist['epoch'], hist['val_mae'], label=['测试误差'])
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.xlabel('训练次数')
    plt.ylabel('均方误差[$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label=['训练误差'])
    plt.plot(hist['epoch'], hist['val_mse'], label=['测试误差'])
    plt.ylim([0, 20])
    plt.legend()
    plt.show()
plot_history(history) #把平均绝对误差和均方误差图画出来

#5．模型验证
#测试集测试泛化模型效果
loss,mae,mse=model.evaluate(normed_test_data,test_labels,verbose=2)
print("测试集的平均价绝对误差是：{:5.2f} MPG".format(mae))
#预测验证
test_predictions=model.predict(normed_test_data).flatten()
#画图表示
plt.scatter(test_labels,test_predictions)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.xlabel('真实值[MPG]')
plt.ylabel('预测值[MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100,100],[-100,100])
plt.show()