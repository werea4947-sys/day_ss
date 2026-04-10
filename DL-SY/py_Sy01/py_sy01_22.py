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

# 从array 转到 Image
def array_to_img(array):
        array = array * 255
        new_img = Image.fromarray(array.astype(np.uint8))
        return new_img


reshaped_imgs = [img.reshape(28, 28) for img in origin_7_imgs]

#使用np.Hstack水平连接图像和垂直使用np.vstack
comb_imgs = np.vstack([np.hstack(reshaped_imgs[i:i+10]) for i in range(0, 100, 10)])
# 效果图
plt.imshow(comb_imgs, cmap='gray')
plt.axis('off')  #隐藏轴
plt.show()
#主成分

from sklearn.decomposition import PCA

def Aaa(data_mat, top_n_feat=99999999):
    pca = PCA(n_components=top_n_feat)
    pca.fit(data_mat)  # Fit适配
    # 使用适配好的PCA模型将输入的 data_mat 转换为低维空间
    low_d_data_mat = pca.transform(data_mat)  # 转换数据
    #从低维重构数据
    recon_mat_for_7_imgs = pca.inverse_transform(low_d_data_mat)

    return recon_mat_for_7_imgs
#指定数量为1
recon_mat_for_7_imgs = Aaa(np.array(origin_7_imgs), 1)
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





