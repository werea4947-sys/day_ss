#引入 MNIST 数据集、numpy 和 PIL 的 Image
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#获得 MNIST 数据集的所有图片和标签
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
imgs = mnist.train.images
labels = mnist.train.labels
print(type(imgs)) # <type 'numpy.ndarray'>
print(type(labels)) # <type 'numpy.ndarray'>
print(imgs.shape) # (60000, 784)
print(labels.shape) # (60000,)
#取前 1000 张图片里的 100 张数字是 7 的图片
origin_7_imgs = []
for i in range(1000):
    if labels[i] == 7 and len(origin_7_imgs) < 100:
        origin_7_imgs.append(imgs[i])
print(np.array(origin_7_imgs).shape)  # (100, 784)
'''
# 从array 转到 Image
def array_to_img(array):
        array = array * 255
        new_img = Image.fromarray(array.astype(np.uint8))
        return new_img
#拼图
def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
    new_img = Image.new(new_type, (col * each_width, row * each_height))
    for i in range(len(origin_imgs)):
        each_img = array_to_img(np.array(origin_imgs[i]).reshape(each_width, each_height))
        new_img.paste(each_img, ((i % col) * each_width, (i // col) * each_height))
    return new_img'''

reshaped_imgs = [img.reshape(28, 28) for img in origin_7_imgs]
#使用np.Hstack水平连接图像和垂直使用np.vstack
comb_imgs = np.vstack([np.hstack(reshaped_imgs[i:i+10]) for i in range(0, 100, 10)])
# 效果图
plt.imshow(comb_imgs, cmap='gray')
plt.axis('off')  #隐藏轴
plt.show()


#主成分
def pca(data_mat, top_n_feat=99999999):
    # 获取数据条数和每条的维数
    num_data, dim = data_mat.shape
    print(num_data)  # 100
    print(dim)  # 784
    # 数据中心化，即指变量减去它的均值
    mean_vals = data_mat.mean(axis=0)  # shape:(784,)
    mean_removed = data_mat - mean_vals  # shape:(100, 784)
    # 计算协方差矩阵（Find covariance matrix）
    cov_mat = np.cov(mean_removed, rowvar=0)  # shape：(784, 784)
    # 计算特征值(Find eigenvalues and eigenvectors)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))# 计算特征值和特征向量，shape 分别为（784，）和(784, 784)
    eig_val_index = np.argsort(eig_vals)  # 对特征值进行从小到大排序，argsort 返回的是索引，即下标
    eig_val_index = eig_val_index[:-(top_n_feat + 1) : -1] # 最大的前 top_n_feat 个特征的索引
    # 取前 top_n_feat 个特征后重构的特征向量矩阵 reorganize eig vects,
    # shape 为(784, top_n_feat)，top_n_feat 最大为特征总数
    reg_eig_vects = eig_vects[:, eig_val_index]
    # 将数据转到新空间
    low_d_data_mat = mean_removed * reg_eig_vects  # shape: (100, top_n_feat), top_n_feat 最大为特征总数
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals  # 根据前几个特征向量重构回去的矩阵，shape:(100, 784)
    return low_d_data_mat, recon_mat
#调用 PCA 进行降维
low_d_feat_for_7_imgs, recon_mat_for_7_imgs = pca(np.array(origin_7_imgs), 1)# 只取最重要的 1 个特征
print(low_d_feat_for_7_imgs.shape) # (100, 1)
print(recon_mat_for_7_imgs.shape) # (100, 784)
#看降维后只用 1 个特征向量重构的效果图
'''low_d_img = comb_imgs(recon_mat_for_7_imgs, 10, 10, 28, 28, 'L')
low_d_img.show()'''
#出图
# 获取复数数组的实部
magnitude_imgs = [np.real(img) for img in recon_mat_for_7_imgs]
# 将每个图像重塑为(28, 28)
reshaped_imgs = [img.reshape(28, 28) for img in magnitude_imgs]
# 使用np.Hstack水平连接图像和垂直使用np.vstack
low_d_img = np.vstack([np.hstack(reshaped_imgs[i:i+10]) for i in range(0, 100, 10)])
# 展示效果图
plt.imshow(low_d_img, cmap='gray')
plt.axis('off')  # 隐藏轴
plt.show()





