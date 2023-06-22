import time
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

with open('faceDR', 'r') as f:
    for line in f:
        print(line)

label = []
num_delete = []
num = np.zeros([4, 5])

with open('faceDR', 'r') as f:
    for line in f.readlines():
        m = re.findall(r' (\w+)', line)
        if len(m) > 2:
            m = m[0:5]
            m[0] = int(m[0])
            label.append(m)


            gender_dict = {'female': 0, 'male': 1}
            m[1] = gender_dict.get(m[1], -1)
            num[0][m[1]] += 1

            age_dict = {'child': 0, 'teen': 1, 'adult': 2, 'senior': 3}
            m[2] = age_dict.get(m[2], -1)
            num[1][m[2]] += 1

            skin_dict = {'white': 0, 'black': 1, 'hispanic': 2, 'asian': 3, 'other': 4}
            m[3] = skin_dict.get(m[3], -1)
            num[2][m[3]] += 1

            expression_dict = {'smiling': 0, 'funny': 1, 'serious': 2}
            m[4] = expression_dict.get(m[4], -1)
            num[3][m[4]] += 1

        else:
            num_delete.append(int(m[0]))


skin_colors = ['white', 'black', 'hispanic', 'asian', 'other']
num_skin_colors = num[2]
plt.bar(skin_colors, num_skin_colors)
plt.xlabel('Skin Color')
plt.ylabel('Count')
plt.title('Distribution of Skin Colors')
plt.show()

features = ['Gender', 'Age', 'Skin Color', 'Expression']
num_features = len(features)

data = pd.read_csv('faceR', header=None, sep='\s+', engine='python')
data[0:10]


idx = [np.where(data[0] == i)[0][0] for i in num_delete]
data = data.drop(idx)

x = data.iloc[:, 1:].to_numpy(dtype='float32')
label = np.array(label)
y = label[:, 3]

scaler = preprocessing.MinMaxScaler()
x_normalized = scaler.fit_transform(x)

pca = PCA(n_components=50)
x_pca = pca.fit_transform(x_normalized)

np.save('test_data.npy', x)
np.save('test_label.npy', label)

#“”“KNN邻近算法与SVM支持向量机 交叉验证”“”
X_train, X_test, y_train, y_test = train_test_split(x_pca, y, random_state=2)
knn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5)

grid_search_knn.fit(X_train, y_train)
best_knn = grid_search_knn.best_estimator_
best_params_knn = grid_search_knn.best_params_

knn_cv_accuracy = grid_search_knn.best_score_

svm = SVC()
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_
best_params_svm = grid_search_svm.best_params_

svm_cv_accuracy = grid_search_svm.best_score_

knn_accuracy = best_knn.score(X_test, y_test)
svm_accuracy = best_svm.score(X_test, y_test)

knn_y_pred = best_knn.predict(X_test)
svm_y_pred = best_svm.predict(X_test)

print("KNN准确率：", knn_accuracy)
print("SVM准确率：", svm_accuracy)

model_names = ['KNN', 'SVM']
accuracies = [knn_accuracy, svm_accuracy]
plt.bar(model_names, accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of KNN and SVM Accuracies')
plt.show()
plt.plot(model_names, accuracies, marker='o')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of KNN and SVM Accuracies')
plt.show()
