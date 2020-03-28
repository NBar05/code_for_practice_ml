import pandas as pd
import numpy as np
from sklearn.svm import SVC

# импортируем данные и присваиваем к переменным
data = pd.read_csv("/Users/nikitabaramiya/Desktop/ML/svm-data.csv", header=None)

X = data.loc[:, 1:]
y = data.loc[:, 0]

# обучаем классификатор с линейным ядром, параметром C=100000 и random_state=241
svc = SVC(C=100000, random_state=241, kernel='linear')
svc.fit(X, y)

# находим номера объектов, являющихся опорными, и записываем в текствой файл
with open('hw3.1.txt', 'w') as f:
       for i in range(len(svc.support_)):
           f.write(str(svc.support_[i] + 1))
           if(i != len(svc.support_) - 1):
               f.write(" ")
