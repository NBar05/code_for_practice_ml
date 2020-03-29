import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# импорт данных, бинаризация столбца с полом, присваивание к переменным
data = pd.read_csv("/Users/nikitabaramiya/Desktop/ML/abalone.csv", header=0)

data["Sex"] = data["Sex"].map(lambda x: 1 if x == "M" else (-1 if x == "F" else 0))

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# создаём генератор
kfold = KFold(n_splits=5, random_state=1, shuffle=True)
# тренируем лес с заданным числом деревьев, сохраняем среднее значение ошибки
a = []
for i in range(50):
    a.append(np.mean(cross_val_score(RandomForestRegressor(random_state=1, n_estimators=i+1), \
    X, y, cv=kfold, scoring='r2')))
    print(i)

# найдём минимальное количество деревьев, при котором качество на кросс-валидации выше 0.52
b = np.array(a)
print(a.index(b[np.around(b, 2) > 0.52][0]) + 1)
