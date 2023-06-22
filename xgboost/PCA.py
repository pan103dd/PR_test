import numpy as np
import cv2
import math
import os
from PIL import Image

imageMatrix = []

file_path = './rawdata'
folders = os.listdir(file_path)
for file in folders:
    file_name = file_path + '/' +file
    with open(file_name, 'rb') as f:
        content = f.read()
    data = np.frombuffer(content, dtype=np.uint8)

    # 图片转为统一尺寸
    img_resized = cv2.resize(data.reshape(int(math.sqrt(data.shape[0])), int(math.sqrt(data.shape[0]))), (128, 128),
                             interpolation=cv2.INTER_CUBIC)
    mats = np.array(img_resized)
    # cv2.imshow('real', img_resized)
    # cv2.waitKey(0)
    # print(data)

    imageMatrix.append(mats.ravel())

imageMatrix = np.array(imageMatrix, dtype=object)  # imageMatrix是图片矩阵
print(imageMatrix)

# 矩阵转置后每一列都是一个图像,对行求均值
imageMatrix = np.transpose(imageMatrix)
imageMatrix = np.mat(imageMatrix)

# 原始矩阵的行均值
mean_img = np.mean(imageMatrix, axis=1)

# 此处用于显示平均脸，如果想要存储到本地，可以自主添加文件存储代码
mean_img1 = np.reshape(mean_img, (int(math.sqrt(mean_img.shape[0])), int(math.sqrt(mean_img.shape[0]))))
im = Image.fromarray(np.uint8(mean_img1))
im.show()


# 均值中心化
imageMatrix = imageMatrix - mean_img

# W是特征向量， V是特征向量组 (3458 X 3458)
imag_mat = (imageMatrix.T * imageMatrix) / float(len(folders))
W, V = np.linalg.eig(imag_mat)
# V_img是协方差矩阵的特征向量组
V_img = imageMatrix * V

# 降序排序后的索引值
axis = W.argsort()[::-1]
V_img = V_img[:, axis]

number = 0
x = sum(W)
for i in range(len(axis)):
    number += W[axis[i]]
    if float(number) / x > 0.9:# 取累加有效值为0.9
        print('累加有效值是：', i) # 前62个特征值保存大部分特征信息
        break
# 取前62个最大特征值对应的特征向量，组成映射矩阵
V_img_finall = V_img[:, :62]


# 降维后的训练样本空间
projectedImage = V_img_finall.T * imageMatrix
np.savetxt('pca.csv', projectedImage, delimiter=',')

