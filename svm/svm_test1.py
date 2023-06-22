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

            # 性别
            gender_dict = {'female': 0, 'male': 1}
            m[1] = gender_dict.get(m[1], -1)
            num[0][m[1]] += 1

            # 年龄
            age_dict = {'child': 0, 'teen': 1, 'adult': 2, 'senior': 3}
            m[2] = age_dict.get(m[2], -1)
            num[1][m[2]] += 1

            # 种族（肤色）
            skin_dict = {'white': 0, 'black': 1, 'hispanic': 2, 'asian': 3, 'other': 4}
            m[3] = skin_dict.get(m[3], -1)
            num[2][m[3]] += 1

            # 表情
            expression_dict = {'smiling': 0, 'funny': 1, 'serious': 2}
            m[4] = expression_dict.get(m[4], -1)
            num[3][m[4]] += 1

        else:
            num_delete.append(int(m[0]))


# 统计
skin_colors = ['white', 'black', 'hispanic', 'asian', 'other']
num_skin_colors = num[2]
# 绘图
plt.bar(skin_colors, num_skin_colors)
plt.xlabel('Skin Color')
plt.ylabel('Count')
plt.title('Distribution of Skin Colors')
plt.show()
plt.pie(num_skin_colors, labels=skin_colors, autopct='%1.1f%%')
plt.title('Distribution of Skin Colors')
plt.axis('equal')
plt.show()

features = ['Gender', 'Age', 'Skin Color', 'Expression']
num_features = len(features)
feature_counts = num.sum(axis=1)
plt.bar(features, feature_counts)
plt.xlabel('Features')
plt.ylabel('Count')
plt.title('Feature Count')
plt.show()
plt.plot(features, feature_counts, marker='o', linestyle='-', color='b')
plt.xlabel('Features')
plt.ylabel('Count')
plt.title('Feature Count')
plt.show()

data = pd.read_csv('faceR', header=None, sep='\s+', engine='python')
data[0:10]

missing_data = data.isnull().mean()
sns.heatmap(data.isnull(), cmap='YlGnBu')
plt.title('Missing Data Heatmap')
plt.show()
plt.bar(missing_data.index, missing_data)
plt.xlabel('Features')
plt.ylabel('Missing Data Percentage')
plt.title('Missing Data Percentage by Feature')
plt.xticks(rotation=45)
plt.show()

idx = [np.where(data[0] == i)[0][0] for i in num_delete]
data = data.drop(idx)

x = data.iloc[:, 1:].to_numpy(dtype='float32')
label = np.array(label)
y = label[:, 3]

scaler = preprocessing.MinMaxScaler()
x_normalized = scaler.fit_transform(x)

selected_feature = x_normalized[:, 0]
plt.hist(selected_feature, bins=20)
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.title('Histogram - After Normalization')
plt.show()
plt.boxplot(selected_feature)
plt.xlabel('Features')
plt.ylabel('Value')
plt.title('Boxplot - After Normalization')
plt.show()

pca = PCA(n_components=50)
x_pca = pca.fit_transform(x_normalized)

plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot - PCA')
plt.show()
sns.heatmap(np.corrcoef(x_pca.T), cmap='YlOrRd')
plt.xlabel('Principal Components')
plt.ylabel('Principal Components')
plt.title('Correlation Heatmap - PCA')
plt.show()

np.save('test_data.npy', x)
np.save('test_label.npy', label)

"""
“”“KNN邻近算法与SVM支持向量机 交叉验证”“”
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
"""

start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(x_pca, y, random_state=2)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_accuracy = knn.score(X_test, y_test)
print("KNN Accuracy:", knn_accuracy)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
svm_accuracy = clf.score(X_test, y_test)
print("SVM Accuracy:", svm_accuracy)

end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")

knn_y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
knn_confusion_matrix = confusion_matrix(y_test, knn_y_pred)
print("KNN Confusion Matrix:")
print(knn_confusion_matrix)

svm_y_pred = clf.predict(X_test)

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

svm_confusion_matrix = confusion_matrix(y_test, svm_y_pred)
print("SVM Confusion Matrix:")
print(svm_confusion_matrix)

knn_confusion_matrix = confusion_matrix(y_test, knn_y_pred)
sns.heatmap(knn_confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - KNN')
plt.show()

knn_confusion_matrix = confusion_matrix(y_test, knn_y_pred)
df_confusion = pd.DataFrame(knn_confusion_matrix, index=np.unique(y_test), columns=np.unique(y_test))
df_confusion.plot(kind='bar', stacked=True, cmap='Blues')
plt.xlabel('True Labels')
plt.ylabel('Count')
plt.title('Confusion Matrix - KNN')
plt.legend(title='Predicted Labels')
plt.show()

from sklearn.metrics import precision_score, recall_score
knn_precision = precision_score(y_test, knn_y_pred, average='macro')
knn_recall = recall_score(y_test, knn_y_pred, average='macro')
print("KNN Precision:", knn_precision)
print("KNN Recall:", knn_recall)

svm_precision = precision_score(y_test, svm_y_pred, average='macro')
svm_recall = recall_score(y_test, svm_y_pred, average='macro')
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)

knn_y_prob = knn.predict_proba(X_test)
svm_y_prob = clf.decision_function(X_test)
n_classes = len(np.unique(y_test))
plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test, knn_y_prob[:, i], pos_label=i)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Class {} (AUC = {:.2f})'.format(i, roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curves')
plt.legend(loc="lower right")
plt.show()

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test, svm_y_prob[:, i], pos_label=i)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Class {} (AUC = {:.2f})'.format(i, roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curves')
plt.legend(loc="lower right")
plt.show()

knn_precision, knn_recall, _ = precision_recall_curve(y_test, knn_y_prob[:, 1], pos_label=1)
plt.plot(knn_recall, knn_precision, marker='.', label='KNN')
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_y_prob[:, 1], pos_label=1)
plt.plot(svm_recall, svm_precision, marker='.', label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

knn_f1 = f1_score(y_test, knn_y_pred, average='macro')
print("KNN F1 Score:", knn_f1)
svm_f1 = f1_score(y_test, svm_y_pred, average='macro')
print("SVM F1 Score:", svm_f1)


