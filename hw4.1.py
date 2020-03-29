import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

# импорт данных
data_train = pd.read_csv("/Users/nikitabaramiya/Desktop/ML/salary-train.csv", header=0)
data_test = pd.read_csv("/Users/nikitabaramiya/Desktop/ML/salary-test-mini.csv", header=0)

X_train = data_train.iloc[:, 0:3]
y_train = data_train.iloc[:, -1]
X_test = data_test.iloc[:, 0:3]

# заменим всё, кроме букв и цифр, на пробелы
X_train['FullDescription'] = X_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
X_test['FullDescription'] = X_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

# заполняем пропуски
X_train['LocationNormalized'].fillna('nan', inplace=True)
X_train['ContractTime'].fillna('nan', inplace=True)

# приводим всё к нижнему регистру
X_train['FullDescription'] = X_train['FullDescription'].str.lower()
X_test['FullDescription'] = X_test['FullDescription'].str.lower()

# считаем TF-IDF для столбца с текстом
tvectorizer = TfidfVectorizer(min_df = 5)
X_train_vec = tvectorizer.fit_transform(X_train['FullDescription'])
X_test_vec = tvectorizer.transform(X_test['FullDescription'])

# применяем one-hot кодирование для остальных столбцов
dvectorizer = DictVectorizer()
X_train_categ = dvectorizer.fit_transform(X_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = dvectorizer.transform(X_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

# соединяем разреженные матрицы горизонтально
X_for_train = hstack([X_train_vec, X_train_categ])
X_for_test = hstack([X_test_vec, X_test_categ])

# тренируем гребневую регрессию
ridge = Ridge(alpha=1, random_state=241)
ridge.fit(X_for_train, y_train)

# предсказываем значения для тренировочной выборки
print([round(i, 2) for i in ridge.predict(X_for_test)])
