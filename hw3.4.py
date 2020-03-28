import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

# импорт даннных
data1 = pd.read_csv("data/classification.csv", header=0)
data2 = pd.read_csv("data/scores.csv", header=0)

# функция для определния характеристик confusion matrix
def function(a, b):
    sum = 0
    for i in range(data1.shape[0]):
        if data1.true[i] == a and data1.pred[i] == b:
            sum += 1
    return sum

# считаем confusion matrix
TP = function(1, 1)
FP = function(0, 1)
FN = function(1, 0)
TN = function(0, 0)
# и печатаем их
print([TP, FP, FN, TN])

# считаем основные метрики качества классификатора
print(round(accuracy_score(data1.true, data1.pred), 3))
print(round(precision_score(data1.true, data1.pred), 3))
print(round(recall_score(data1.true, data1.pred), 3))
print(round(f1_score(data1.true, data1.pred), 3))

# считаем ROC-AUC для каждого алгоритма
print(round(roc_auc_score(data2.true, data2.score_logreg), 3))
print(round(roc_auc_score(data2.true, data2.score_svm), 3))
print(round(roc_auc_score(data2.true, data2.score_knn), 3))
print(round(roc_auc_score(data2.true, data2.score_tree), 3))

# находим классификатор достигающий наибольшей точности (Precision)
# при полноте (Recall) не менее 70%
a = precision_recall_curve(data2.true, data2.score_logreg)
print(round(a[0][a[1] >= 0.7].max(), 2))
b = precision_recall_curve(data2.true, data2.score_svm)
print(round(b[0][b[1] >= 0.7].max(), 2))
c = precision_recall_curve(data2.true, data2.score_knn)
print(round(c[0][c[1] >= 0.7].max(), 2))
d = precision_recall_curve(data2.true, data2.score_tree)
print(round(d[0][d[1] >= 0.7].max(), 2))

# делаем графическую иллюстрацию предыдущего поиска
for algorithm in data2.columns[1:]:
    yscores = data2[algorithm]
    precision, recall, thresholds = precision_recall_curve(data2.true, yscores)
    plt.plot(recall, precision, label = algorithm)
    plt.xlim(0.7,1)
    plt.ylim(0.5, 0.7)
    plt.legend()

plt.show()
