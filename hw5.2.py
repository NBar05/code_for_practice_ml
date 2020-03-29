import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

# импорт данных, разбиение на обучающие и тестовые выборки
data = pd.read_csv("/Users/nikitabaramiya/Desktop/ML/gbm-data.csv", header=0)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
minimums = []
index = []
for q in [1, 0.5, 0.3, 0.2, 0.1]:
    # тренируем градиентный бустинг
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=q)
    clf.fit(X_train, y_train)
    
    # метод staged_decision_function - для предсказания качества 
    # на обучающей и тестовой выборке на каждой итерации;
    # подставляем y_pred в логистическую функцию;
    # считаем log-loss
    train_score = []
    test_score = []
    for y_predicted in clf.staged_decision_function(X_train):
        train_score.append(log_loss(y_train, 1 / (1 + np.exp(-y_predicted))))
    for y_predicted in clf.staged_decision_function(X_test):
        test_score.append(log_loss(y_test, 1 / (1 + np.exp(-y_predicted))))
    
    # находим минимальное значение log-loss на тестовой выборке и номер итерации,
    # на котором оно достигается, при learning_rate = 0.2
    minimums.append(np.min(test_score))
    index.append(test_score.index(np.min(test_score)))
    
    # строим график значений log-loss на обучающей и тестовой выборках
    plt.figure()
    plt.plot(test_score, 'r', linewidth=2)
    plt.plot(train_score, 'g', linewidth=2)
    plt.legend(['test', 'train'])

plt.show()

# находим наилучшее качество и соотвествующее количество итераций
a = minimums.index(np.min(minimums))
i = index[a]

# смотрим на качество леса при количестве деревьев = количеству итераций
rfc = RandomForestClassifier(n_estimators=i, random_state=241)
rfc.fit(X_train, y_train)
print(round(log_loss(y_test, rfc.predict_proba(X_test)), 2))
