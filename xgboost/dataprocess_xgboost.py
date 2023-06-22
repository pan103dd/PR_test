import numpy as np
import re
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split  # 分割数据模块

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# with open('faceDR','r') as f:
#     for line in f.readlines():
#         print(line)

label = []  # 存取转换为数据的类别卷标
num_delete = []  # 缺失数据
num = np.zeros([4, 5])  # 统计每个特征的数量

with open('faceDR', 'r') as f:
    for line in f.readlines():
        m = re.findall(r' (\w+)', line)
        if len(m) > 2:
            m = m[0:5]
            m[0] = int(m[0])
            # 判断性别
            if m[1] == 'female':
                m[1] = 0
                num[0][0] = num[0][0] + 1
            elif m[1] == 'male':
                m[1] = 1
                num[0][1] = num[0][1] + 1
            # 判断年龄
            if m[2] == 'child':
                m[2] = 0
                num[1][0] = num[1][0] + 1
            elif m[2] == 'teen':
                m[2] = 1
                num[1][1] = num[1][1] + 1
            elif m[2] == 'adult':
                m[2] = 2
                num[1][2] = num[1][2] + 1
            elif m[2] == 'senior':
                m[2] = 3
                num[1][3] = num[1][3] + 1
            # 判断肤色
            if m[3] == 'white':
                m[3] = 0
                num[2][0] = num[2][0] + 1
            elif m[3] == 'black':
                m[3] = 1
                num[2][1] = num[2][1] + 1
            elif m[3] == 'hispanic':
                m[3] = 2
                num[2][2] = num[2][2] + 1
            elif m[3] == 'asian':
                m[3] = 3
                num[2][3] = num[2][3] + 1
            elif m[3] == 'other':
                m[3] = 4
                num[2][4] = num[2][4] + 1
            # 判断表情
            if m[4] == 'smiling':
                m[4] = 0
                num[3][0] = num[3][0] + 1
            elif m[4] == 'funny':
                m[4] = 1
                num[3][1] = num[3][1] + 1
            elif m[4] == 'serious':
                m[4] = 2
                num[3][2] = num[3][2] + 1

            label.append(m)
        else:
            num_delete.append(int(m[0]))

# print(num)
# print(label)
# print(num_delete)

# import matplotlib.pyplot as plt
#
# plt.figure(1)
# x1_pic = ['female', 'male']
# y1_pic = [num[0][0], num[0][1]]
# plt.bar(x1_pic, y1_pic, color=plt.cm.coolwarm(np.linspace(0, 1, len(x1_pic))))
# # 设置图形标题和坐标轴标签
# plt.title('sex_num')
# plt.xlabel('sex')
# plt.ylabel('num')
#
# plt.figure(2)
# x2_pic = ['child', 'teen', 'adult', 'senior']
# y2_pic = [num[1][0], num[1][1], num[1][2], num[1][3]]
# plt.bar(x2_pic, y2_pic, color=plt.cm.coolwarm(np.linspace(0, 1, len(x2_pic))))
# # 设置图形标题和坐标轴标签
# plt.title('age_num')
# plt.xlabel('age')
# plt.ylabel('num')
#
# # 显示图形
# plt.show()

# =====================================================
import pandas as pd

data = pd.read_csv('faceR', header=None, sep='\s+', engine='python')

# 删除无效数据
idx = []
for i in num_delete:
    idx.append(np.where(data[0] == i)[0][0])
print('idx:', idx)
data = data.drop(idx)

x = data.iloc[:, 1:].values.astype('float32')
# x = preprocessing.normalize(x, norm='l2')

label = np.array(label)
y = label[:, 1]

# 保存数据
np.save('test_data', x)
np.save('test_label', label)

# # knn
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=2)
# # 建立模型
# knn = KNeighborsClassifier()
# # 训练模型
# knn.fit(X_train, y_train)
# # 将准确率打印出
# print(knn.score(X_test, y_test))

# # 支持向量机
# clf1 = xgb(n_estimators = 100)
# clf1.fit(x, y)
# y_pred = clf1.predict(X_test)
# acc = (y_pred == y_test).sum()/len(data) * 100
# print('accuracy:', acc)
#
# miss_classified = (y_pred != y_test).sum()
# print("MissClassified: ", miss_classified)

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(acc)
    print(tip + '正确率：\t', float(acc.sum()) / a.size)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

# 数据转换为数据矩阵类DMatrix
data_train = xgb.DMatrix(X_train, label=y_train)
data_test = xgb.DMatrix(X_test, label=y_test)

# 设置参数
param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 4}
watchlist = [(data_test, 'eval'), (data_train, 'train')]

# 训练模型
bst = xgb.train(param, data_train, num_boost_round=4, evals=watchlist)

# 测试
y_hat = bst.predict(data_test)
show_accuracy(y_hat, y_test, 'XGBoost ')
