import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Импорт данных, отбор переменных и удаление строк с пустыми ячейками
titanic = pd.read_csv("data/titanic.csv", header=0)
data = titanic.loc[:, ["Pclass", "Fare", "Age", "Sex", "Survived"]].dropna()

# Делим данные на объясняющие и объясняемую
X = data.loc[:, ["Pclass", "Fare", "Age", "Sex"]]
y = data.loc[:, "Survived"]

# Бинаризуем столбец пола
X.Sex = pd.get_dummies(X.Sex)["female"]

# Инициализируем алгоритм и прогоняем наши данные
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

# Смотрим на важность признаков
importances = clf.feature_importances_
print(importances)
