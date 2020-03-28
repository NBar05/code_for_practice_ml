import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# импортируем данные и преобразуем их в матричный формат, попутно деля на X и y
test = pd.read_csv("data/perceptron-test.csv", header=None)
train = pd.read_csv("data/perceptron-train.csv", header=None)

X_train = np.array(train.loc[:, 1:])
y_train = np.array(train.loc[:, 0])
X_test = np.array(test.loc[:, 1:])
y_test = np.array(test.loc[:, 0])

# масштабируем по тренировочной выборке
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# обучаем персептрон
clf1 = Perceptron(random_state=241)
clf2 = Perceptron(random_state=241)
clf1.fit(X_train, y_train)
clf2.fit(X_train_scaled, y_train)

# предсказываем значения для тестовой выборки
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test_scaled)

# смотрим прирост качества при стандартизации
score1 = accuracy_score(y_test, y_pred1)
score2 = accuracy_score(y_test, y_pred2)
print(round(score2 - score1, 3))

# обучаем персептрон с доп ограничениями
clf3 = Perceptron(random_state=241, max_iter=5, tol=None)
clf4 = Perceptron(random_state=241, max_iter=5, tol=None)
clf3.fit(X_train, y_train)
clf4.fit(X_train_scaled, y_train)

# предсказываем
pred3 = clf3.predict(X_test)
pred4 = clf4.predict(X_test_scaled)

# смотрим
score3 = accuracy_score(y_test, pred3)
score4 = accuracy_score(y_test, pred4)
print(round(score4 - score3, 3))
