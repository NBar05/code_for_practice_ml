import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# импорт данных, отбор переменных и удаление строк с пустыми ячейками
titanic = pd.read_csv("data/titanic.csv", header=0)
data = titanic.loc[:, ["Pclass", "Fare", "Age", "Sex", "Survived"]].dropna()

# делим данные на объясняющие и объясняемую
X = data.loc[:, ["Pclass", "Fare", "Age", "Sex"]]
y = data.loc[:, "Survived"]

# бинаризуем столбец пола
X.Sex = pd.get_dummies(X.Sex)["female"]

# инициализируем алгоритм и прогоняем наши данные
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

# смотрим на важность признаков
importances = clf.feature_importances_
print(importances)
