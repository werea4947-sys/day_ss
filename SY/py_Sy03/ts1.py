import tensorflow.examples.tutorials.mnist.input_data as input_data

#获得 MNIST 数据集的所有图片和标签
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
#显示一下minst数据集的类型
print("MNIST数据集的类型是： %s'" % (type(mnist)))
print("训练集的数量是：%d" % mnist.train.num_examples)
print("验证集的数量是：%d" % mnist.validation.num_examples)
print("测试集的数量是：%d" % mnist.test.num_examples)

#训练集和验证集划分
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels
'''print("Type of training is %s" % (type(train_img )))
print("Type of trainlabel is %s" % (type(train_label )))
print("Type of testing is %s" % (type(test_img )))
print("Type of testing is %s" % (type(test_label )))'''

#神经网络
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(solver='sgd',hidden_layer_sizes=(100, 100), max_iter=25, alpha=1e-5, verbose=15, tol=1e-5, learning_rate_init=0.1)
#hidden_layer_sizes: tuple，length = n_layers - 2，默认值（100，）第i个元素表示第i个隐藏层中的神经元数量。
#verbose:bool，可选，默认为False 是否将进度消息打印到stdout。
#max_iter   :optional，默认值200。最大迭代次数。
#alpha: float，可选，默认为0.0001。L2惩罚（正则化项）参数。
#tol:float，optional，默认1e-4 优化的容忍度，容差优化
#random_state:int，RandomState实例或None，可选，默认无随机数生成器的状态或种子
#learning_rate_init:double，可选，默认为0.001。使用初始学习率。它控制更新权重的步长。仅在solver ='sgd’或’adam’时使用。
mlp.fit(train_img,train_label)
rv=mlp.score(train_img,train_label)
print("模型精度是:",rv)
#训练精度
rs=mlp.score(test_img,test_label)
print("训练精度是:", rs)