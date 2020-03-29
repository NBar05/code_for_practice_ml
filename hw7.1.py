import time
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# посчитаем, сколько времени нужно компьютеру на выполнение всех этих команд
main_start_time = datetime.datetime.now()

# извлекаю данные
Xy = pd.read_csv('data/data7/features.csv', index_col='match_id')
y_train = Xy.radiant_win

X_test = pd.read_csv('data/data7/features_test.csv', index_col='match_id')

# удаляем лишние признаки (связаны с итогами матчами)
features = ["duration", "radiant_win", "tower_status_radiant", "tower_status_dire", "barracks_status_radiant", "barracks_status_dire"]
X_train = Xy.drop(features, axis=1)

# получаем имена столбцов с пропущенными значениями
print(X_train.count()[X_train.count() != X_train.shape[0]].index)
# заполняем пустые значения нулями и в тренировачной и в тестовой (потом понадобится)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# используем для разбиения данных на кросс-валидации
kfold = KFold(n_splits=5, shuffle=True, random_state=241)

# оцениваем качество градиентного бустинга с помощью кросс-валидации, используя при этом разное количество деревьев
# попутно оцениваем время проведения кросс-валидации для каждого количества деревьев
for i in [10, 20, 30]:
    start_time = datetime.datetime.now()
    print(str(i) + ": " + str(round(np.mean(cross_val_score(GradientBoostingClassifier(n_estimators=i), \
    X_train, y_train, cv=kfold, scoring='roc_auc')), 3)))
    print('Time elapsed: ' + str(datetime.datetime.now() - start_time))

# стандартизируем признаки для следующего метода
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# оцениваем качество логистической регрессии с помощью кросс-валидации, используя при этом разные C
# попутно оцениваем время проведения кросс-валидации для каждого C
for C in np.power(10.0, np.arange(-5, 6)):
    start_time = datetime.datetime.now()
    print(str(C) + ": " + str(round(np.mean(cross_val_score(LogisticRegression(solver='lbfgs', C=C, penalty='l2', max_iter=10000), \
    X_train_scaled, y_train, cv=kfold, scoring='roc_auc')), 3)))
    print('Time elapsed: ' + str(datetime.datetime.now() - start_time))

## другой способ с применением кросс-валидации и вложенным в неё грид-поиском по параметрам
##grid = {'C': np.power(10.0, np.arange(-5, 6))}
##gs = GridSearchCV(LogisticRegression(solver='lbfgs', penalty='l2'), grid, scoring='roc_auc', cv=kfold)
##print(np.mean(cross_val_score(gs, X_train_scaled, y_train, cv=kfold, scoring='roc_auc')))

# удаляем лишние признаки (категориальные) и снова стандартизируем
X_train_new = X_train.drop(["lobby_type", "r1_hero", "r2_hero", "r3_hero", "r4_hero", "r5_hero", \
"d1_hero", "d2_hero", "d3_hero", "d4_hero", "d5_hero"], axis=1)
X_test_new = X_test.drop(["lobby_type", "r1_hero", "r2_hero", "r3_hero", "r4_hero", "r5_hero", \
"d1_hero", "d2_hero", "d3_hero", "d4_hero", "d5_hero"], axis=1)

X_train_newscaled = scaler.fit_transform(X_train_new)
X_test_newscaled = scaler.fit_transform(X_test_new)

# оцениваем качество логистической регрессии на обновлённых данных с помощью кросс-валидации, используя при этом разные C
# заодно считаем время
for C in np.power(10.0, np.arange(-5, 6)):
    start_time = datetime.datetime.now()
    print(str(C) + ": " + str(round(np.mean(cross_val_score(LogisticRegression(solver='lbfgs', C=C, penalty='l2', max_iter=10000), \
    X_train_newscaled, y_train, cv=kfold, scoring='roc_auc')), 3)))
    print('Time elapsed: ' + str(datetime.datetime.now() - start_time))

# извлекаем используемых героев
a = []
b = []
for i in ["r1_hero", "r2_hero", "r3_hero", "r4_hero", "r5_hero", "d1_hero", "d2_hero", "d3_hero", "d4_hero", "d5_hero"]:
    a = a + X_train[i].unique().tolist()
    b = b + X_test[i].unique().tolist()

# число используемых героев на тренировачной выборке
N = pd.DataFrame(set(a)).shape[0]
print("Number of heroes: " + str(N))

# создаём и заполняем матрицы для мешка слов
X_pick_train = np.zeros((X_train_newscaled.shape[0], max(set(a))))
X_pick_test = np.zeros((X_test_newscaled.shape[0], max(set(b))))

for i, match_id in enumerate(X_train.index):
    for p in range(5):
        X_pick_train[i, X_train.loc[match_id, 'r{}_hero'.format(p+1)] - 1] = 1
        X_pick_train[i, X_train.loc[match_id, 'd{}_hero'.format(p+1)] - 1] = -1

for i, match_id in enumerate(X_test.index):
    for p in range(5):
        X_pick_test[i, X_test.loc[match_id, 'r{}_hero'.format(p+1)] - 1] = 1
        X_pick_test[i, X_test.loc[match_id, 'd{}_hero'.format(p+1)] - 1] = -1

# объединяем данные
X_train_updated = np.hstack([X_train_newscaled, X_pick_train])
X_test_updated = np.hstack([X_test_newscaled, X_pick_test])

# оцениваем качество логистической регрессии на дополненных данных с помощью кросс-валидации, используя при этом разные C
# заодно считаем время
for C in np.power(10.0, np.arange(-5, 6)):
    start_time = datetime.datetime.now()
    print(str(C) + ": " + str(round(np.mean(cross_val_score(LogisticRegression(solver='lbfgs', C=C, penalty='l2', max_iter=10000), \
    X_train_updated, y_train, cv=kfold, scoring='roc_auc')), 3)))
    print('Time elapsed: ' + str(datetime.datetime.now() - start_time))

# считаем итоговую модель и получаем предсказания модели
clf = LogisticRegression(solver='lbfgs', C=0.01, penalty='l2', max_iter=10000)
clf.fit(X_train_updated, y_train)
y_pred_proba = clf.predict_proba(X_test_updated)

# минимальное и максимальное значение прогноза на тестовой выборке
print(round(pd.DataFrame(y_pred_proba)[0].max(), 3), round(pd.DataFrame(y_pred_proba)[0].min(), 3))

# создаём файл с предсказаниями
data = {"match_id": np.array(X_test.index), "radiant_win": y_pred_proba[:, 0]}
pd.DataFrame(data).to_csv('data/data7/result_data.csv', encoding='utf-8', index=False)

# итоговое время
print('Total time: ' + str(datetime.datetime.now() - main_start_time))
