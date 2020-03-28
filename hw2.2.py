import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# импортируем данные с разными характеристиками, определяющие стоимость жилья
boston = load_boston()

# масштабируем признаки, создаём переменные признаков и предсказываемой переменной
X =  scale(boston.data)
y = boston.target

# создаёем генератор для кросс-валидации
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
# прогоняем алгоритм с разным параметром метрики Минковского
a = []
b = []
for p in np.linspace(1, 10, 200):
    a.append(np.mean(cross_val_score(KNeighborsRegressor(n_neighbors=5, weights='distance', p=p), \
    X, y, cv=kfold, scoring='neg_mean_squared_error')))
    b.append(p)

# печатаем лучший параметр метрики Минковского
print(b[a.index(np.max(a))])
