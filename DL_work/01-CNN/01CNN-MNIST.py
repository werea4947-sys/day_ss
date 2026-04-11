import os

# Workaround for Windows OpenMP duplicate runtime conflicts (libiomp5md.dll).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
import swanlab
import matplotlib.pyplot as plt

#超参数
BATCH_SIZE = 64 #每批数据的大小
EPOCHS = 15 #训练轮数
LR = 0.0015 #学习率
data_root = Path(__file__).resolve().parent / 'MNIST_data'
train_path = str(data_root) #训练数据路径
test_path = str(data_root) #测试数据路径

#加载mnist数据集 train：60000张训练图像 test：10000张测试图像
#训练数据集
train_data = torchvision.datasets.MNIST(
    root=train_path, #数据集存放路径
    train=True, #是否为训练集
    transform=torchvision.transforms.ToTensor(), #将图像转换为Tensor
    download=True,
)

#测试数据集
test_data = torchvision.datasets.MNIST(
    root=test_path, #数据集存放路径
    train=False, #是否为测试集
    transform=torchvision.transforms.ToTensor(), #将图像转换为Tensor
    download=True,
)

#分批训练 sample_size = 64 channel_size = 1 图像大小为28*28 （64，1，28，28）
#torch.utils.data.DataLoader()函数可以将数据集分成小批量进行训练
train_loader = DataLoader(
    dataset=train_data, #数据集
    batch_size=BATCH_SIZE, #每批数据的大小
    shuffle=True, #是否打乱数据
)
test_loader = DataLoader(
    dataset=test_data, #数据集
    batch_size=BATCH_SIZE, #每批数据的大小
    shuffle=False, #是否打乱数据
)

#定义卷积神经网络模型
#nn.Module是所有神经网络模块的基类，您的模型应该继承这个类，并实现forward方法
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() #调用父类的构造函数
        #建立第一个卷积-激活-池化层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, #输入通道数为1 因为mnist数据集是灰度图像
                out_channels=16, #输出通道数为16
                kernel_size=3, #卷积核大小为3*3
                stride=1, #步长为1
                padding=1), #填充为1
                #输出特征图的大小为(64, 16, 28, 28) 
            nn.ReLU(), #ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2) #最大池化层
        )#输出特征图的大小为(64, 16, 14, 14)

        #卷积层2 输入通道数为16 输出通道数为32 卷积核大小为3*3 步长为1 填充为1
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, #输入通道数为16
                out_channels=32, #输出通道数为32
                kernel_size=3, #卷积核大小为3*3
                stride=1, #步长为1
                padding=1), #填充为1
                #输出特征图的大小为(64, 32, 14, 14) 
            nn.ReLU(), #ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2) #最大池化层
        )#输出特征图的大小为(64, 32, 7, 7)

        #全连接层 输入特征数为32*7*7 输出特征数为10
        self.fc = nn.Linear(in_features=32*7*7, out_features=10)

    #前向传播函数
    def forward(self, x):
        #卷积层1
        x = self.conv1(x)
        #卷积层2
        x = self.conv2(x)
        #输出特征图的大小为(64, 32, 7, 7)
        #将特征图展平为一维向量
        x = x.view(x.size(0), -1)#64*32*7*7,view函数将x的形状改变为(64, 32*7*7) 
        #全连接层
        output= self.fc(x)
        return output
    
#实例化模型
cnn = CNN()

#优化器和损失函数
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR) #Adam优化器

#初始化swanlab实验记录
run = swanlab.init(
    project="mnist-cnn",
    experiment_name="cnn-loss-tracking",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
    },
)

#训练模型
for epoch in range(EPOCHS):
    cnn.train() #设置模型为训练模式
    epoch_loss = 0.0
    #batch_idx是批次索引 x_in是输入数据 y_label是标签
    for batch_idx, (x_in, y_label) in enumerate(train_loader):       
        optimizer.zero_grad() #每个batch前清空梯度，避免梯度累计
        output = cnn(x_in) #前向传播
        loss = criterion(output, y_label) #计算损失
        loss.backward() #反向传播
        optimizer.step() #更新参数
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    swanlab.log({"train/loss": avg_loss, "epoch": epoch + 1})

    print('Epoch: ', epoch+1, 'Loss: {:.4f}'.format(avg_loss)) #打印损失

torch.save(cnn.state_dict(), 'cnn_mnist.pth') #保存模型参数

#调用模型评估函数
def evaluate(model, test_loader):
    model.eval() #设置模型为评估模式
    correct = 0 #正确预测的数量
    total = 0 #总的样本数量
    all_targets = []
    all_predictions = []
    with torch.no_grad(): #不计算梯度
        for data, target in test_loader:
            output = model(data) #前向传播
            _, predicted = torch.max(output.data, 1) #获取预测结果
            total += target.size(0) #更新总的样本数量
            correct += (predicted == target).sum().item() #更新正确预测的数量
            all_targets.append(target)
            all_predictions.append(predicted)

    accuracy = correct / total #计算准确率
    print('Test Accuracy: {:.2f}%'.format(accuracy * 100)) #打印准确率

    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
    for t, p in zip(targets, predictions):
        confusion_matrix[t.long(), p.long()] += 1

    print('Confusion Matrix:')
    print(confusion_matrix)
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(7, 6))
    plt.imshow(confusion_matrix.numpy(), interpolation='nearest', cmap='Blues')
    plt.title('MNIST Confusion Matrix')
    plt.colorbar()
    tick_marks = list(range(10))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def classify_one_image(model, dataset, index=0):
    model.eval()
    image, true_label = dataset[index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted_label = int(torch.argmax(output, dim=1).item())

    print(f'Single Image Classification -> index: {index}, true: {true_label}, predicted: {predicted_label}')

    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze(0).numpy(), cmap='gray')
    plt.title(f'True: {true_label} | Pred: {predicted_label}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

cm = evaluate(cnn, test_loader)
plot_confusion_matrix(cm)
classify_one_image(cnn, test_data, index=0)

run.finish()