# 先导入库
from keras.models import Model
import keras.layers as KL
import keras.backend as K

# %% 建立resnet模型（提取特征）
def buiding_block(filter_nums, block_nums):
   if block_nums != 0:
       stride_step = 1  # reset中当非第一个block的模型中stride为1
   else:
       stride_step = 2  # 第一个block中stris为2，降低图像尺寸咯

   def blocks(x):
       y = KL.Conv2D(filter_nums, (1, 1), strides=stride_step)(x)
       y = KL.BatchNormalization(axis=-1)(y)
       y = KL.Activation('sigmoid')(y)

       y = KL.Conv2D(filter_nums, (3, 3), padding='same')(y)
       y = KL.BatchNormalization(axis=-1)(y)
       y = KL.Activation('sigmoid')(y)

       y = KL.Conv2D(filter_nums * 4, (1, 1))(y)
       y = KL.BatchNormalization(axis=-1)(y)

       # 判断 shortcut 的加入方式
       if block_nums == 0:
           shortcut = KL.Conv2D(filter_nums * 4, (1, 1), strides=stride_step)(x)
           shortcut = KL.BatchNormalization(axis=-1)(shortcut)

       else:
           shortcut = x

       y = KL.Add()([y, shortcut])  # 使用KL.add()函数将输出的数据相加，不增加filter。
       # 而若使用concatenate()则会将输出层增加filter的量，若使用merge对网络层进行合并模式{“sum”，“mul”，“concat”，“ave”，“cos”，“dot”}
       y = KL.Activation('sigmoid')(y)
       return y

   return blocks
# 以上是建立上图的两种不同的building block 与定义类非常相似。

# %%定义模型

def resnet_me(inputs):
   x = KL.Conv2D(64, (11, 11), strides=5, padding='same')(inputs)
   x = KL.BatchNormalization(axis=-1)(x)
   x = KL.Activation('sigmoid')(x)

   filter_nums = 32
   block_nums = [3, 3]  # 这里的block_nums 是blocks的数量，可以做很深的如5层的[3,4,5,3,3]resnet
   for i, blockss in enumerate(block_nums):
       for block_n in range(blockss):
           x = buiding_block(filter_nums, block_n)(x)
           # filter_nums *=2
   x = KL.UpSampling2D((2, 2))(x)  # 这里做一个上采样，为了Dense的时候降维
   x = KL.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
   x = KL.Flatten()(x)
   x = KL.Dense(10, activation='softmax')(x)
   return x

input_tensor1 = KL.Input((28, 28, 1)) #定义输入 Mnist数据为28*28 的灰度图像
out = resnet_me(input_tensor1)  # 输出为resnet_me的结果
model = Model([input_tensor1],[out])  #初始化模型
model.summary() #summary看下输出有没有错误

#画图展示模型结构
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import plot_model
plot_model(model,to_file='Resnet.png',show_shapes=True)
ls=mpimg.imread('Resnet.png')
plt.imshow(ls)#显示图片
plt.axis('off')#不显示坐标轴
plt.show()

import  tensorflow as tf
from  keras import datasets, layers, optimizers,models
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(tf.__version__)

def preprocess(x, y):
    # [b, 28, 28], [b]
    print(x.shape,y.shape)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28,28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y

#导入mnist数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

# 数据预处理
#训练集
batchsz = 128
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000)
train_db = train_db.batch(batchsz)
train_db = train_db.map(preprocess)
train_db = train_db.repeat(2)
#输出一下训练集的shape
x,y = next(iter(train_db)) #iter()可迭代生成对象  next()
print('train sample:', x.shape, y.shape)

#测试集
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batchsz).map(preprocess)
test_db = test_db.repeat(2)
#输出测试集的shape
x_test,y_test = next(iter(test_db))
print('test sample:',x_test.shape,y_test.shape)

#训练模型
model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
NUM_EPOCHS = 5
history = model.fit(train_db,validation_data=(test_db),epochs=NUM_EPOCHS)
#
#模型验证
loss,accuracy=model.evaluate(test_db,batch_size=12)
print('Test loss:',loss)
print('Test accuracy:',accuracy)
#model = tf.keras.models.load_model(model_path)
predict = model.predict(x_test) #使用测试集测试结果
