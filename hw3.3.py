import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# импортируем данные, присваиваем переменные
data = pd.read_csv("data/data-logistic.csv", header=None)

X = data.loc[:, 1:]
y = data.loc[:, 0]

# функция для определения весов логистической регрессии градиентным спуском 
def gradient_descent(w1, w2, k, C, max_iter):
    for i in range(max_iter):
        w1new = w1 + k * np.mean(y * X[1] * (1 - 1/(1 + np.exp(-y * (w1*X[1] + w2*X[2])))) ) - k * C * w1
        w2new = w2 + k * np.mean(y * X[2] * (1 - 1/(1 + np.exp(-y * (w1*X[1] + w2*X[2])))) ) - k * C * w2
        if ((w1 - w1new)**2 + (w2 - w2new)**2 )**(1/2) <= 1e-5:
            return [w1new, w2new]
            break
        else:
            w1 = w1new
            w2 = w2new

a1, a2 = gradient_descent(0, 0, 0.1, 0, 10000)
b1, b2 = gradient_descent(0, 0, 0.1, 10, 10000)

# оцениваем вероятности, подсчитанные алгоритмом
# сигмоидная функция
def predictions(w1, w2):
    pred = []
    for i in range(X.shape[0]):
        pred.append(1 / (1 + np.exp(-w1*X[1][i]-w2*X[2][i]) ) )
    return pred

# считаем ROC-AUC
print(round(roc_auc_score(y, predictions(a1, a2)), 3))
print(round(roc_auc_score(y, predictions(b1, b2)), 3))
