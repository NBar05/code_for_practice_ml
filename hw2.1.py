import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# импортируем данные 
data = pd.read_csv("/Users/nikitabaramiya/Desktop/ML/wine.data", header=None)

# разделяем данные на признаки и переменную с классами
y = np.array(data.loc[:, 0])
X = np.array(data.loc[:, 1:])
# масштабируем признаки (нулевое м.о. и единичная дисперсия)
X_scale = scale(X)

# генератор разбиений
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# считаем усреднённую долю верных ответов для каждого количества соседей
a = []
for k in range(1,51):
    a.append(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = k), \
    X, y, cv = kfold)))

# находим наилучшее качество и соответствующее количество соседей
print(np.max(a))
print(a.index(np.max(a))+1)

# считаем усреднённую долю верных ответов для каждого количества соседей
b = []
for k in range(1,51):
    b.append(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors = k), \
    X_scale, y, cv = kfold)))

# находим наилучшее качество и соответствующее количество соседей
print(np.max(b))
print(b.index(np.max(b))+1)
