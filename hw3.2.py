import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# импортируме данные
data = fetch_20newsgroups(
                      subset='all',
                      categories=['alt.atheism', 'sci.space']
                          )

# Считаем TF-IDF и присваиваем к переменным данные
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data.data)
y = data.target

# настраиваем параметры, среди которых ищем самый оптимальный
grid = {'C': np.power(10.0, np.arange(-5, 6))}
# настраиваем генератор для кросс-валидации
cv = KFold(n_splits=5, shuffle=True, random_state=241)
# и классификатор, для которого ищем оптимум
clf = SVC(kernel='linear', random_state=241)

# запускаем поиск
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

# тренируем на оптимальном С
svc = SVC(C=1, kernel='linear', random_state=241)
svc.fit(X, y)

# извлекаем веса слов
coef = svc.coef_
q = pd.DataFrame(coef.toarray()).transpose()

# сортируем и получаем индексы самых весомых слов
top10 = abs(q).sort_values([0], ascending=False).head(10)
indices = top10.index

# находим слова по индексам
words = []
features = vectorizer.get_feature_names()
for i in indices:
    words.append(features[i])

# печатаем самые важные слова
print(",".join(sorted(words)))
