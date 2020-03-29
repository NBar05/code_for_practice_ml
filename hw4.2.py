import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# импорт данных
close_prices = pd.read_csv("data/close_prices.csv", header=0)
djia_index = pd.read_csv("data/djia_index.csv", header=0)

# применяем метод главных компонент
pca = PCA(n_components=10)
pca.fit(close_prices.iloc[:, 1:], djia_index.iloc[:, -1])

# смотрим, какую часть дисперсии объясняют 4 компоненты
print(np.sum(pca.explained_variance_ratio_[:4]))

# преобразуем данные с помощью метода главных компонент и берём первую компоненту
transforming = pd.DataFrame(pca.transform(close_prices.iloc[:, 1:]))[0]

# смотрим корреляцию первой компоненты и индекса Доу-Джонса
print(round(np.corrcoef(transforming, djia_index.iloc[:, -1])[0, 1], 2))

# смотрим веса компаний в первой компоненте, находим самую весомую
first_comp = pca.components_[0]
index = list(first_comp).index(np.max(abs(first_comp)))
print(close_prices.columns[index+1])

# записываем в файл
f = open('hw4.2.3.txt', 'w')
f.write(str(close_prices.columns[index+1]))
